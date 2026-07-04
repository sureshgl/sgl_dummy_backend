# tests/test_e2e.py
import subprocess, time, requests, signal, os

def test_generate():
    env = {**os.environ, "SGLANG_PLATFORM": "dummy"}
    # proc = subprocess.Popen([
    #     "python", "-m", "sglang.launch_server",
    #     "--model-path", "facebook/opt-125m",
    #     "--disable-cuda-graph", "--tp-size", "1", "--port", "30001",
    # ], env=env)
    
    # time.sleep(60)  # wait for startup
    
    try:
        resp = requests.post("http://localhost:30000/generate", json={
            "text": "The capital of France is",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0}
        })
        assert resp.status_code == 200
        assert "text" in resp.json()
        print ("Hello, world! Response text:", resp.json()["text"])
        print ("Response:", resp.json())
    finally:
        print ("Cleaning up server process...")
        # proc.send_signal(signal.SIGTERM)
        # proc.wait()