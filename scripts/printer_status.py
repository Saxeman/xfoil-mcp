"""Read-only printer check: connect over the local network and report status.

Moves nothing. Needs LAN Only Mode and Developer Mode on the printer, and
the three XFOIL_PRINTER_* variables set (see printer.env, never committed).
"""

import os
import sys
import time

import bambulabs_api as bl

KEYS = ("XFOIL_PRINTER_IP", "XFOIL_PRINTER_ACCESS_CODE", "XFOIL_PRINTER_SERIAL")
ip, code, serial = (os.environ.get(k) for k in KEYS)
if not all((ip, code, serial)):
    sys.exit(f"set {', '.join(KEYS)} (source printer.env)")

printer = bl.Printer(ip, code, serial)
printer.mqtt_start()                     # status and commands only; no camera

for _ in range(20):
    if printer.mqtt_client_ready():
        break
    time.sleep(0.5)
else:
    sys.exit("no connection: check LAN Only + Developer Mode, the IP, and the access code")

time.sleep(2)                            # let the first status report arrive
print("state:       ", printer.get_state())
print("print status:", printer.get_current_state())
print("bed:         ", printer.get_bed_temperature())
print("nozzle:      ", printer.get_nozzle_temperature())
printer.disconnect()
