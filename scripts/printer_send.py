"""Send one sliced file to the printer and start it. Moves the machine.

Usage: printer_send.py <file.gcode.3mf> <ams-slot 1-4>

Refuses unless the printer is idle, shows exactly what will be sent, and
requires typing PRINT. Then confirms the print really started, because the
library's upload logs failures instead of raising them.
"""

import hashlib
import os
import sys
import time
from pathlib import Path
import json
# import logging
# logging.basicConfig(level=logging.DEBUG)

import bambulabs_api as bl
from bambulabs_api.states_info import GcodeState

READY = {GcodeState.IDLE, GcodeState.FINISH, GcodeState.FAILED}
STARTED = {GcodeState.PREPARE, GcodeState.RUNNING}

if len(sys.argv) != 3:
    sys.exit("usage: printer_send.py <file.gcode.3mf> <ams-slot 1-4>")
path, slot = Path(sys.argv[1]), int(sys.argv[2])
if not path.name.endswith(".gcode.3mf") or not path.exists():
    sys.exit(f"{path}: need an existing .gcode.3mf from Bambu Studio's 'Export plate sliced file'")
if slot not in (1, 2, 3, 4):
    sys.exit("AMS slot must be 1 to 4")

printer = bl.Printer(os.environ["XFOIL_PRINTER_IP"], os.environ["XFOIL_PRINTER_ACCESS_CODE"],
                     os.environ["XFOIL_PRINTER_SERIAL"])
printer.mqtt_start()
for _ in range(20):
    if printer.mqtt_client_ready():
        break
    time.sleep(0.5)
else:
    sys.exit("no connection to the printer")
time.sleep(2)

state = printer.get_state()
data = path.read_bytes()
print(f"file:     {path.name} ({len(data) / 1024:.0f} KB, sha256 {hashlib.sha256(data).hexdigest()[:12]})")
print(f"printer:  {state.name}, bed {printer.get_bed_temperature()} °C, nozzle {printer.get_nozzle_temperature()} °C")
print(f"filament: AMS slot {slot}")
if state not in READY:
    printer.disconnect()
    sys.exit(f"printer is {state.name}; it must be idle or finished before a new print")

if input("Plate clear and filament loaded? Type PRINT to start: ").strip() != "PRINT":
    printer.disconnect()
    sys.exit("not sent")

# The library's FTPS client closes the data connection without ending TLS
# first; the printer then reports the upload as broken (426). Ending TLS
# properly fixes it; the timeout stops that step from ever hanging forever.
printer.ftp_client.ftps.unwrap = True
printer.ftp_client.ftps.timeout = 30

uploaded = printer.upload_file(open(path, "rb"), path.name)
if not uploaded or "No file" in str(uploaded):
    printer.disconnect()
    sys.exit(f"upload failed (library returned {uploaded!r}); nothing was started")
print("uploaded:", uploaded)

# The library's start_print can't request a timelapse, so build the same
# command it sends, plus "timelapse". Reaching into its internals
# (_client, command_topic) is fragile: recheck if the library is upgraded.
command = {"print": {
    "command": "project_file",
    "param": "Metadata/plate_1.gcode",
    "file": path.name,
    "url": f"ftp:///{path.name}",
    "bed_type": "textured_plate",
    "bed_leveling": True,
    "flow_cali": True,
    "vibration_cali": True,
    "layer_inspect": False,
    "use_ams": True,
    "ams_mapping": [slot - 1],
    "timelapse": True,
    "sequence_id": "10000001",
}}
mqtt = printer.mqtt_client
sent = mqtt._client.publish(mqtt.command_topic, json.dumps(command))
sent.wait_for_publish(timeout=10)
if not sent.is_published():
    printer.disconnect()
    sys.exit("the start command was not delivered to the printer")

for _ in range(30):                       # up to a minute for the state to change
    time.sleep(2)
    state = printer.get_state()
    if state in STARTED:
        print(f"started: printer is {state.name}")
        break
else:
    print(f"warning: start command accepted, but the printer still reports {state.name}")
printer.disconnect()
