"""Raw connection check: log in to the printer's status channel and ask for one report.

Read-only. Prints the login result, then one status report if the printer answers.
"""

import json
import os
import ssl
import time

import paho.mqtt.client as mqtt

ip = os.environ["XFOIL_PRINTER_IP"]
code = os.environ["XFOIL_PRINTER_ACCESS_CODE"]
serial = os.environ["XFOIL_PRINTER_SERIAL"]


def on_connect(client, userdata, flags, reason, properties=None):
    print("login result:", reason)
    if not reason.is_failure:
        client.subscribe(f"device/{serial}/report")
        client.publish(f"device/{serial}/request",
                       json.dumps({"pushing": {"sequence_id": "0", "command": "pushall"}}))


def on_message(client, userdata, msg):
    report = json.loads(msg.payload).get("print", {})
    print("report:", {k: report.get(k) for k in ("gcode_state", "bed_temper", "nozzle_temper")})
    client.disconnect()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set("bblp", code)
# The printer uses a self-signed certificate. Skipping verification is fine for
# this local check; a production client would pin the printer's certificate.
client.tls_set(cert_reqs=ssl.CERT_NONE)
client.tls_insecure_set(True)
client.on_connect = on_connect
client.on_message = on_message
client.connect(ip, 8883, 60)
client.loop_start()
time.sleep(10)
client.loop_stop()
print("done")
