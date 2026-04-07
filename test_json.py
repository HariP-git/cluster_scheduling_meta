import urllib.request
import json
import time
import subprocess

# Start server
proc = subprocess.Popen(["uv", "run", "server", "--port", "7860"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

try:
    # Reset
    req = urllib.request.Request('http://127.0.0.1:7860/reset', data=b'', method='POST')
    reset_resp = json.loads(urllib.request.urlopen(req).read().decode())
    print("RESET:")
    print(list(reset_resp["observation"].keys()))

    # Step 1
    req1 = urllib.request.Request('http://127.0.0.1:7860/step', data=json.dumps({"action": {"stage_id": 1}}).encode(), headers={'Content-Type': 'application/json'}, method='POST')
    step1 = json.loads(urllib.request.urlopen(req1).read().decode())
    print("\nSTEP 1 keys inside observation:")
    print(list(step1["observation"].keys()))

finally:
    proc.terminate()
