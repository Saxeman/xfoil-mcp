"""The printer backend: slice the approved part, send it to a Bambu printer, start it.

Used only when XFOIL_PRINT_BACKEND=bambu; otherwise every print goes to the
dry backend's outbox. It runs after the print queue has checked the
approval, so it never decides whether to print, only how.

Each step demands positive evidence, because every tool underneath reports
failure quietly: the slicer can write a project with no toolpaths in it (or
with generic settings, if its presets weren't resolved), the printer library
logs a failed upload instead of raising, and a start command can be
delivered without the printer ever starting.

Configuration, from the environment:
    XFOIL_PRINTER_IP, XFOIL_PRINTER_ACCESS_CODE, XFOIL_PRINTER_SERIAL
    XFOIL_PRINTER_AMS_SLOT   1-4: the slot holding the filament the settings are for
    XFOIL_SLICER_SETTINGS    folder with machine.json, process.json and filament.json,
                             made by scripts/flatten_profile.py (default: <repo>/slicer)
    XFOIL_SLICER             Bambu Studio's executable (default: the macOS app)
    XFOIL_SLICER_PROFILES    Bambu Studio's profiles/BBL folder, where the printer's
                             model id is looked up (default: inside the macOS app)
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
import subprocess
import time
import hashlib
from typing import Callable

from xfoil_mcp.printing import DryBackend, PrintRequest

from contextlib import suppress
from xml.etree import ElementTree
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_APP = Path("/Applications/BambuStudio.app/Contents")
SETTINGS_FILES = ("machine.json", "process.json", "filament.json")
TOOLPATHS = "Metadata/plate_1.gcode"       # what the start command tells the printer to run
READY = {"IDLE", "FINISH", "FAILED"}       # states a new print may start from
STARTED = {"PREPARE", "RUNNING"}           # evidence the print really began
REQUIRED = ("XFOIL_PRINTER_IP", "XFOIL_PRINTER_ACCESS_CODE", "XFOIL_PRINTER_SERIAL",
            "XFOIL_PRINTER_AMS_SLOT")

# The start command declares the plate; it must match the slicer settings'
# curr_bed_type, which check_settings enforces.
BED_TYPES = {"Textured PEI Plate": "textured_plate"}


class PrinterError(Exception):
    """A step of sending a print failed. failure_kind says who has to act."""

    def __init__(self, message: str, failure_kind: str = "infrastructure"):
        super().__init__(message)
        self.failure_kind = failure_kind


@dataclass(frozen=True)
class PrinterConfig:
    ip: str
    access_code: str
    serial: str
    ams_slot: int
    settings: Path
    slicer: Path
    profiles: Path
    slice_timeout_s: float = 180.0
    start_timeout_s: float = 60.0

    @classmethod
    def from_env(cls, env=os.environ) -> PrinterConfig:
        """Read and check the configuration. ValueError names what is wrong."""
        missing = [key for key in REQUIRED if not env.get(key)]
        if missing:
            raise ValueError(f"the printer backend needs {', '.join(missing)}")
        if env["XFOIL_PRINTER_AMS_SLOT"] not in {"1", "2", "3", "4"}:
            raise ValueError("XFOIL_PRINTER_AMS_SLOT must be 1, 2, 3 or 4")
        settings = Path(env.get("XFOIL_SLICER_SETTINGS", _REPO / "slicer"))
        absent = [name for name in SETTINGS_FILES if not (settings / name).is_file()]
        if absent:
            raise ValueError(f"{settings} is missing {', '.join(absent)}; "
                             "make them with scripts/flatten_profile.py")
        slicer = Path(env.get("XFOIL_SLICER", _APP / "MacOS/BambuStudio"))
        if not slicer.is_file():
            raise ValueError(f"no slicer at {slicer}; set XFOIL_SLICER")
        config = cls(ip=env["XFOIL_PRINTER_IP"], access_code=env["XFOIL_PRINTER_ACCESS_CODE"],
                     serial=env["XFOIL_PRINTER_SERIAL"], ams_slot=int(env["XFOIL_PRINTER_AMS_SLOT"]),
                     settings=settings, slicer=slicer,
                     profiles=Path(env.get("XFOIL_SLICER_PROFILES", _APP / "Resources/profiles/BBL")))
        check_settings(config)
        return config


def _setting(config: PrinterConfig, file: str, key: str):
    """One setting from one settings file. Per-filament settings are lists."""
    value = json.loads((config.settings / file).read_text()).get(key)
    return value[0] if isinstance(value, list) and value else value


def check_settings(config: PrinterConfig) -> None:
    """Refuse settings files that would slice wrongly without saying so."""
    for name in SETTINGS_FILES:
        if "inherits" in json.loads((config.settings / name).read_text()):
            # The command-line slicer doesn't resolve inheritance: it fills
            # every missing setting with generic defaults (60 mm/s, "PLA").
            raise ValueError(f"{name} still inherits from another preset; "
                             "flatten it with scripts/flatten_profile.py")
    bed = _setting(config, "process.json", "curr_bed_type")
    if bed not in BED_TYPES:
        raise ValueError(f"process.json's curr_bed_type is {bed!r}; this backend supports "
                         f"{', '.join(BED_TYPES)}")

def check_sliced(path: Path, config: PrinterConfig) -> dict:
    """Positive evidence the sliced file is printable and is what was intended.

    A file existing proves nothing: it must contain toolpaths, for the
    filament the settings name, with a real time and weight (a weight of 0
    is how unresolved presets showed up), for a known printer model.
    """
    with zipfile.ZipFile(path) as sliced:
        names = set(sliced.namelist())
        if TOOLPATHS not in names or sliced.getinfo(TOOLPATHS).file_size == 0:
            raise PrinterError("the slicer wrote a file with no toolpaths in it")
        info = ElementTree.fromstring(sliced.read("Metadata/slice_info.config"))
    meta = {m.get("key"): m.get("value") for m in info.iter("metadata")}
    materials = [f.get("type") for f in info.iter("filament")]
    wanted = _setting(config, "filament.json", "filament_type")
    if materials != [wanted]:
        raise PrinterError(f"the sliced file is for {materials}, but the settings say {wanted}")
    seconds, grams = float(meta.get("prediction") or 0), float(meta.get("weight") or 0)
    if seconds <= 0 or grams <= 0:
        raise PrinterError(f"the slicer estimated {seconds:.0f} s and {grams:.1f} g; "
                           "the settings are probably incomplete")
    model_id = meta.get("printer_model_id") or _fill_model_id(path, config)
    return {"material": wanted, "estimated_minutes": round(seconds / 60, 1),
            "estimated_grams": round(grams, 1), "printer_model_id": model_id}


def _fill_model_id(path: Path, config: PrinterConfig) -> str:
    """Write the printer model id the command-line slicer leaves blank.

    The GUI looks it up from the printer's model file by name; this does the
    same lookup. Only the file's description changes: the toolpaths and
    their checksum are copied byte for byte.
    """
    model = _setting(config, "machine.json", "printer_model")
    try:
        model_id = json.loads((config.profiles / "machine" / f"{model}.json").read_text())["model_id"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        raise PrinterError(f"could not find the model id for {model!r}: {exc}") from exc
    blank = b'key="printer_model_id" value=""'
    temporary = path.with_name(path.name + ".tmp")
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as out:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "Metadata/slice_info.config":
                data = data.replace(blank, f'key="printer_model_id" value="{model_id}"'.encode())
            out.writestr(item, data)
    shutil.move(temporary, path)
    return model_id

def slice_part(stl: Path, out: Path, config: PrinterConfig) -> dict:
    """Slice the approved STL with the committed settings. Returns the evidence.

    The slicer's output is captured: this runs inside the MCP server, whose
    stdout is the protocol stream.
    """
    s = config.settings
    command = [str(config.slicer),
               "--load-settings", f"{s / 'machine.json'};{s / 'process.json'}",
               "--load-filaments", str(s / "filament.json"),
               "--arrange", "1", "--orient", "0", "--slice", "0",
               "--export-3mf", out.name, "--outputdir", str(out.parent), str(stl)]
    try:
        run = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                             text=True, timeout=config.slice_timeout_s)
    except subprocess.TimeoutExpired:
        raise PrinterError(f"slicing took longer than {config.slice_timeout_s:.0f} s") from None
    except OSError as exc:
        raise PrinterError(f"could not run the slicer: {exc}") from exc
    if run.returncode != 0 or not out.is_file():
        tail = " | ".join((run.stderr or run.stdout).strip().splitlines()[-3:])
        raise PrinterError(f"the slicer failed (exit {run.returncode}): {tail or 'no output'}")
    return check_sliced(out, config)

class BambuPrinter:
    """The few printer operations the backend needs, with the library's quirks handled."""

    def __init__(self, config: PrinterConfig):
        import bambulabs_api   # optional dependency: only the printer backend needs it

        self._printer = bambulabs_api.Printer(config.ip, config.access_code, config.serial)

    def connect(self, timeout_s: float = 15.0) -> None:
        self._printer.mqtt_start()                      # status and commands only; no camera
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._printer.mqtt_client_ready() and self.state() != "UNKNOWN":
                return
            time.sleep(0.5)
        raise PrinterError("no status from the printer: check it is on, in LAN Only and "
                           "Developer Mode, and that this app has Local Network permission")

    def state(self) -> str:
        return self._printer.get_state().name

    def upload(self, path: Path) -> None:
        # The library ends the encrypted data connection without closing TLS,
        # which the printer reports as a broken transfer (426); the timeout
        # keeps the proper close from ever hanging.
        self._printer.ftp_client.ftps.unwrap = True
        self._printer.ftp_client.ftps.timeout = 30
        with open(path, "rb") as handle:
            uploaded = self._printer.upload_file(handle, path.name)
        # The library logs upload failures instead of raising them.
        if not uploaded or "No file" in str(uploaded):
            raise PrinterError(f"the upload failed (the library returned {uploaded!r}); "
                               "check the printer has a microSD card with space on it")

    def start(self, filename: str, ams_slot: int, bed_type: str) -> None:
        # The library's start_print can't ask for a timelapse, so this sends
        # the same command it builds, plus "timelapse". It reaches into the
        # library's internals: recheck after upgrading bambulabs_api.
        command = {"print": {
            "command": "project_file", "param": "Metadata/plate_1.gcode",
            "file": filename, "url": f"ftp:///{filename}", "bed_type": bed_type,
            "bed_leveling": True, "flow_cali": True, "vibration_cali": True,
            "layer_inspect": False, "use_ams": True, "ams_mapping": [ams_slot - 1],
            "timelapse": True, "sequence_id": "10000001",
        }}
        mqtt = self._printer.mqtt_client
        sent = mqtt._client.publish(mqtt.command_topic, json.dumps(command))
        sent.wait_for_publish(timeout=10)
        if not sent.is_published():
            raise PrinterError("the start command was not delivered to the printer")

    def close(self) -> None:
        with suppress(Exception):
            self._printer.disconnect()

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PrinterBackend:
    """Slices the approved part, uploads it, starts it, and confirms it started."""

    name = "bambu"

    def __init__(self, outbox: Path, config: PrinterConfig,
                 printer: Callable[[PrinterConfig], BambuPrinter] = BambuPrinter,
                 slicer: Callable[[Path, Path, PrinterConfig], dict] = slice_part,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self._record = DryBackend(outbox)   # every print also leaves the dry run's record
        self._config, self._printer, self._slicer = config, printer, slicer
        self._sleep, self._clock = sleep, clock

    def send(self, request: PrintRequest) -> dict:
        stl = Path(self._record.send(request)["stl_path"])
        sliced = stl.with_name(f"{stl.stem}.gcode.3mf")
        evidence = self._slicer(stl, sliced, self._config)
        result = {
            "backend": self.name, "stl_path": str(stl), "sliced_path": str(sliced),
            "sliced_sha256": _sha256(sliced),
            "slicer_settings": {n: _sha256(self._config.settings / n) for n in SETTINGS_FILES},
            **evidence,
        }
        bed = BED_TYPES[_setting(self._config, "process.json", "curr_bed_type")]
        printer = self._printer(self._config)
        try:
            printer.connect()
            state = printer.state()
            if state not in READY:
                raise PrinterError(f"the printer is {state}; it must be idle or finished first")
            printer.upload(sliced)
            printer.start(sliced.name, self._config.ams_slot, bed)
            result["printer_state"] = self._wait_until_started(printer)
        finally:
            printer.close()
        stl.with_name(f"{stl.stem}.printed.json").write_text(json.dumps(result, indent=2))
        return result

    def _wait_until_started(self, printer: BambuPrinter) -> str:
        deadline = self._clock() + self._config.start_timeout_s
        state = printer.state()
        while state not in STARTED and self._clock() < deadline:
            self._sleep(2.0)
            state = printer.state()
        if state not in STARTED:
            # The command was delivered, so this is not "nothing happened".
            raise PrinterError(f"the start command was delivered, but after "
                               f"{self._config.start_timeout_s:.0f} s the printer still reports "
                               f"{state}. It may yet start: check the printer before printing "
                               "this again")
        return state
