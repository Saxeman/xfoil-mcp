"""The printer backend, with a fake slicer and a fake printer.

Every failure path here is one that really happened, or that the tools
underneath would report quietly.
"""

import json
import zipfile
from pathlib import Path
import stat
import sys
import hashlib

import pytest

from xfoil_mcp import bambu
from xfoil_mcp.cad import SectionPart
from xfoil_mcp.bambu import PrinterConfig, PrinterError, check_sliced, PrinterBackend
from xfoil_mcp.printing import PrintQueue, PrintRefused, PrintRequest

pytest.importorskip("cadquery")   # the print path imports the part builder


GCODE = b"; HEADER_BLOCK_START\nG28\nG1 X10 Y10\n"


def slice_info(material="ABS", prediction="3300", weight="23.1", model_id="") -> bytes:
    """A slice_info.config shaped like Bambu Studio's."""
    return (f'<?xml version="1.0" encoding="UTF-8"?><config><plate>'
            f'<metadata key="printer_model_id" value="{model_id}"/>'
            f'<metadata key="prediction" value="{prediction}"/>'
            f'<metadata key="weight" value="{weight}"/>'
            f'<filament id="1" type="{material}" color="#000000"/>'
            f'</plate></config>').encode()


def write_sliced(path: Path, gcode=GCODE, info=None) -> Path:
    """A fake sliced file; gcode=None leaves out the toolpaths."""
    with zipfile.ZipFile(path, "w") as z:
        if gcode is not None:
            z.writestr("Metadata/plate_1.gcode", gcode)
        z.writestr("Metadata/slice_info.config", info or slice_info())
    return path

@pytest.fixture
def config(tmp_path) -> PrinterConfig:
    """A complete, valid configuration in a temporary folder."""
    settings = tmp_path / "slicer"
    settings.mkdir()
    (settings / "machine.json").write_text(json.dumps({"printer_model": "Bambu Lab X1 Carbon"}))
    (settings / "process.json").write_text(json.dumps({"curr_bed_type": "Textured PEI Plate"}))
    (settings / "filament.json").write_text(json.dumps({"filament_type": ["ABS"]}))
    profiles = tmp_path / "profiles"
    (profiles / "machine").mkdir(parents=True)
    (profiles / "machine" / "Bambu Lab X1 Carbon.json").write_text(json.dumps({"model_id": "BL-P001"}))
    slicer = tmp_path / "BambuStudio"
    slicer.write_text("")
    return PrinterConfig(ip="10.0.0.2", access_code="12345678", serial="00M0", ams_slot=2,
                         settings=settings, slicer=slicer, profiles=profiles)


def env_for(config: PrinterConfig, **changes) -> dict:
    """The environment that describes a config; None removes a variable."""
    env = {"XFOIL_PRINTER_IP": config.ip, "XFOIL_PRINTER_ACCESS_CODE": config.access_code,
           "XFOIL_PRINTER_SERIAL": config.serial, "XFOIL_PRINTER_AMS_SLOT": str(config.ams_slot),
           "XFOIL_SLICER_SETTINGS": str(config.settings), "XFOIL_SLICER": str(config.slicer),
           "XFOIL_SLICER_PROFILES": str(config.profiles)}
    env.update(changes)
    return {k: v for k, v in env.items() if v is not None}


def test_configuration_reads_the_environment(config):
    assert PrinterConfig.from_env(env_for(config)) == config


def test_missing_settings_are_named(config):
    with pytest.raises(ValueError, match="XFOIL_PRINTER_ACCESS_CODE, XFOIL_PRINTER_AMS_SLOT"):
        PrinterConfig.from_env(env_for(config, XFOIL_PRINTER_ACCESS_CODE=None,
                                       XFOIL_PRINTER_AMS_SLOT=None))


def test_the_ams_slot_must_be_one_to_four(config):
    with pytest.raises(ValueError, match="1, 2, 3 or 4"):
        PrinterConfig.from_env(env_for(config, XFOIL_PRINTER_AMS_SLOT="5"))


def test_settings_that_still_inherit_are_refused(config):
    # Unresolved, the command line slices at generic defaults: 60 mm/s, "PLA".
    (config.settings / "filament.json").write_text(json.dumps(
        {"filament_type": ["ABS"], "inherits": "Bambu ABS @base"}))
    with pytest.raises(ValueError, match="filament.json still inherits"):
        PrinterConfig.from_env(env_for(config))


def test_an_unsupported_plate_type_is_refused(config):
    (config.settings / "process.json").write_text(json.dumps({"curr_bed_type": "Cool Plate"}))
    with pytest.raises(ValueError, match="Cool Plate"):
        PrinterConfig.from_env(env_for(config))


def test_a_missing_settings_file_is_named(config):
    (config.settings / "process.json").unlink()
    with pytest.raises(ValueError, match="missing process.json"):
        PrinterConfig.from_env(env_for(config))

def test_a_good_slice_reports_its_estimates(config, tmp_path):
    evidence = check_sliced(write_sliced(tmp_path / "p.gcode.3mf",
                                         info=slice_info(model_id="BL-P001")), config)
    assert evidence == {"material": "ABS", "estimated_minutes": 55.0,
                        "estimated_grams": 23.1, "printer_model_id": "BL-P001"}


def test_a_slice_with_no_toolpaths_is_refused(config, tmp_path):
    with pytest.raises(PrinterError, match="no toolpaths"):
        check_sliced(write_sliced(tmp_path / "p.gcode.3mf", gcode=None), config)


def test_a_slice_for_the_wrong_material_is_refused(config, tmp_path):
    with pytest.raises(PrinterError, match=r"\['PLA'\], but the settings say ABS"):
        check_sliced(write_sliced(tmp_path / "p.gcode.3mf", info=slice_info(material="PLA")), config)


def test_a_zero_weight_estimate_is_refused(config, tmp_path):
    # How unresolved presets first showed up: a 0 g part, with no error.
    with pytest.raises(PrinterError, match="settings are probably incomplete"):
        check_sliced(write_sliced(tmp_path / "p.gcode.3mf", info=slice_info(weight="0")), config)


def test_a_blank_model_id_is_filled_and_the_toolpaths_are_untouched(config, tmp_path):
    path = write_sliced(tmp_path / "p.gcode.3mf")
    evidence = check_sliced(path, config)
    assert evidence["printer_model_id"] == "BL-P001"
    with zipfile.ZipFile(path) as z:
        assert b'value="BL-P001"' in z.read("Metadata/slice_info.config")
        assert z.read("Metadata/plate_1.gcode") == GCODE

def fake_slicer(path: Path, body: str) -> Path:
    """A stand-in executable that receives the real slicer's arguments."""
    path.write_text(f"#!{sys.executable}\nimport sys, zipfile, time\nargs = sys.argv[1:]\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


WRITES_A_SLICE = f"""
out = args[args.index("--outputdir") + 1] + "/" + args[args.index("--export-3mf") + 1]
with zipfile.ZipFile(out, "w") as z:
    z.writestr("Metadata/plate_1.gcode", {GCODE!r})
    z.writestr("Metadata/slice_info.config", {slice_info(model_id="BL-P001")!r})
"""


def test_the_slicer_gets_the_committed_settings_and_no_reorientation(config, tmp_path):
    log = tmp_path / "args.json"
    slicer = fake_slicer(tmp_path / "slicer.py",
                         f"open({str(log)!r}, 'w').write(__import__('json').dumps(args))" + WRITES_A_SLICE)
    config = PrinterConfig(**{**config.__dict__, "slicer": slicer})
    out = tmp_path / "part.gcode.3mf"
    evidence = bambu.slice_part(tmp_path / "part.stl", out, config)
    args = json.loads(log.read_text())
    assert args[args.index("--load-filaments") + 1] == str(config.settings / "filament.json")
    assert args[args.index("--orient") + 1] == "0"
    assert evidence["material"] == "ABS" and out.is_file()


def test_a_failing_slicer_reports_its_last_words(config, tmp_path):
    slicer = fake_slicer(tmp_path / "slicer.py",
                         "sys.stderr.write('run 3002: process not compatible with printer\\n'); sys.exit(3)")
    config = PrinterConfig(**{**config.__dict__, "slicer": slicer})
    with pytest.raises(PrinterError, match="exit 3.*not compatible with printer"):
        bambu.slice_part(tmp_path / "part.stl", tmp_path / "out.gcode.3mf", config)


def test_a_hung_slicer_is_stopped(config, tmp_path):
    slicer = fake_slicer(tmp_path / "slicer.py", "time.sleep(30)")
    config = PrinterConfig(**{**config.__dict__, "slicer": slicer, "slice_timeout_s": 0.5})
    with pytest.raises(PrinterError, match="longer than"):
        bambu.slice_part(tmp_path / "part.stl", tmp_path / "out.gcode.3mf", config)

class FakePrinter:
    """Stands in for BambuPrinter: same five methods, scripted states."""

    def __init__(self, states, upload_error=None):
        self.states, self.upload_error = list(states), upload_error
        self.uploaded, self.started, self.closed = [], [], False

    def __call__(self, config):          # stands in for the BambuPrinter class
        return self

    def connect(self):
        pass

    def state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def upload(self, path):
        if self.upload_error:
            raise PrinterError(self.upload_error)
        self.uploaded.append(path.name)

    def start(self, filename, ams_slot, bed_type):
        self.started.append((filename, ams_slot, bed_type))

    def close(self):
        self.closed = True


def fake_slice(stl, out, config):
    write_sliced(out, info=slice_info(model_id="BL-P001"))
    return check_sliced(out, config)


def request(tmp_path) -> PrintRequest:
    stl = b"solid wing\nendsolid wing\n"
    return PrintRequest(id="req1", design={"label": "4412 flap 10"},
                        part=SectionPart(stl=stl, sha256=hashlib.sha256(stl).hexdigest(), stats={}),
                        created_at=0.0, approved_at=1.0, status="approved")


def backend(tmp_path, config, printer, now=None):
    clock = iter(now or range(1000))
    return PrinterBackend(tmp_path / "outbox", config, printer=printer, slicer=fake_slice,
                          sleep=lambda s: None, clock=lambda: next(clock))


def test_an_approved_part_is_sliced_uploaded_and_started(tmp_path, config):
    printer = FakePrinter(["IDLE", "IDLE", "PREPARE"])
    result = backend(tmp_path, config, printer).send(request(tmp_path))
    sliced = Path(result["sliced_path"])
    assert printer.uploaded == [sliced.name]
    assert printer.started == [(sliced.name, 2, "textured_plate")]
    assert result["printer_state"] == "PREPARE" and result["material"] == "ABS"
    assert result["sliced_sha256"] == hashlib.sha256(sliced.read_bytes()).hexdigest()
    assert set(result["slicer_settings"]) == {"machine.json", "process.json", "filament.json"}
    assert printer.closed
    record = json.loads(sliced.with_name(sliced.name.replace(".gcode.3mf", ".printed.json")).read_text())
    assert record["sliced_sha256"] == result["sliced_sha256"]


def test_a_busy_printer_is_refused_before_anything_is_uploaded(tmp_path, config):
    printer = FakePrinter(["RUNNING"])
    with pytest.raises(PrinterError, match="printer is RUNNING"):
        backend(tmp_path, config, printer).send(request(tmp_path))
    assert printer.uploaded == [] and printer.started == [] and printer.closed


def test_a_failed_upload_sends_no_start_command(tmp_path, config):
    printer = FakePrinter(["IDLE"], upload_error="the upload failed")
    with pytest.raises(PrinterError, match="upload failed"):
        backend(tmp_path, config, printer).send(request(tmp_path))
    assert printer.started == [] and printer.closed


def test_a_start_that_never_starts_says_it_may_yet(tmp_path, config):
    printer = FakePrinter(["IDLE"])
    with pytest.raises(PrinterError, match="was delivered.*may yet start"):
        backend(tmp_path, config, printer, now=[0, 0, 30, 61]).send(request(tmp_path))
    assert printer.started and printer.closed


def test_a_backend_failure_closes_the_request_as_an_infrastructure_failure(tmp_path, config):
    part = request(tmp_path).part
    queue = PrintQueue(backend(tmp_path, config, FakePrinter(["RUNNING"])))
    pending = queue.create({"label": "x"}, part)
    queue.approve(pending.id, part.sha256)
    with pytest.raises(PrintRefused, match="printer is RUNNING.*request the print again") as refused:
        queue.start(pending.id)
    assert refused.value.failure_kind == "infrastructure"
    assert queue.get(pending.id).status == "failed"

def test_the_printer_is_used_only_when_asked_for(tmp_path, config):
    from xfoil_mcp.printing import DryBackend
    from xfoil_mcp.server import _backend

    assert isinstance(_backend({}, tmp_path), DryBackend)
    assert isinstance(_backend(env_for(config, XFOIL_PRINT_BACKEND="bambu"), tmp_path), PrinterBackend)
    with pytest.raises(ValueError, match="'dry' or 'bambu'"):
        _backend({"XFOIL_PRINT_BACKEND": "real"}, tmp_path)
    with pytest.raises(ValueError, match="needs XFOIL_PRINTER_IP"):
        _backend({"XFOIL_PRINT_BACKEND": "bambu"}, tmp_path)
