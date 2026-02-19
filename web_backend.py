from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
import signal
from typing import Dict

app = FastAPI()

origins = [
    "https://fl-frontend-navya.vercel.app"
]

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep track of running processes
processes: Dict[str, subprocess.Popen] = {}

def run_command(name: str, command: list):
    if name in processes and processes[name].poll() is None:
        return {"status": "already running"}
    
    # Log to a file to help debug
    log_file = open(f"{name}.log", "w")
    
    # Prepare environment with venv paths
    env = os.environ.copy()
    venv_bin = os.path.abspath(os.path.join("venv", "Scripts" if os.name == 'nt' else "bin"))
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    
    # On Windows, shell=True with a list can be problematic.
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=log_file,
        shell=True if os.name == 'nt' else False,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0,
        env=env
    )
    processes[name] = process
    return {"status": "started"}

@app.get("/status")
async def get_status():
    status = {}
    for name, proc in processes.items():
        if proc.poll() is None:
            status[name] = "running"
        else:
            status[name] = "idle"
    
    # Ensure all monitored services are in the response
    for service in ["pipeline", "mlflow", "prefect"]:
        if service not in status:
            status[service] = "idle"
    return status

@app.post("/run-pipeline")
async def start_pipeline():
    python_exe = os.path.join("venv", "Scripts", "python.exe") if os.name == 'nt' else os.path.join("venv", "bin", "python")
    return run_command("pipeline", [python_exe, "manage_pipeline.py", "run"])

@app.post("/start-mlflow")
async def start_mlflow():
    python_exe = os.path.join("venv", "Scripts", "python.exe") if os.name == 'nt' else os.path.join("venv", "bin", "python")
    return run_command("mlflow", [python_exe, "manage_pipeline.py", "mlflow"])

@app.post("/start-prefect")
async def start_prefect():
    prefect_exe = os.path.join("venv", "Scripts", "prefect.exe") if os.name == 'nt' else os.path.join("venv", "bin", "prefect")
    return run_command("prefect", [prefect_exe, "server", "start"])

@app.post("/stop/{name}")
async def stop_service(name: str):
    if name in processes and processes[name].poll() is None:
        try:
            if os.name == 'nt':
                # Use taskkill to kill the whole process tree
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(processes[name].pid)], capture_output=True)
            else:
                os.killpg(os.getpgid(processes[name].pid), signal.SIGTERM)
            return {"status": "stopped"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    return {"status": "not running"}

@app.get("/latest-result")
async def get_latest_result():
    reports_dir = "reports"
    if not os.path.exists(reports_dir):
        return {"error": "No reports directory"}
    
    reports = [f for f in os.listdir(reports_dir) if f.endswith(".md")]
    if not reports:
        return {"error": "No reports found"}
    
    # Sort by modification time to get the latest
    latest_report = max([os.path.join(reports_dir, f) for f in reports], key=os.path.getmtime)
    
    try:
        with open(latest_report, "r") as f:
            content = f.read()
            
        import re
        # Much more robust regex: match the label, then skip anything until we find the colon and number
        accuracy_match = re.search(r"(?:Average Final Global Accuracy|Best Accuracy).*?[:\s]+([\d\.]+)", content, re.IGNORECASE)
        rounds_match = re.search(r"Rounds Completed.*?[:\s]+(\d+)", content, re.IGNORECASE)
        
        accuracy = 0
        if accuracy_match:
            val = float(accuracy_match.group(1))
            accuracy = val * 100 if val <= 1.0 else val
            print(f"Parsed Accuracy: {accuracy}% from {accuracy_match.group(1)}")
        else:
            print("Failed to parse accuracy from report")
            
        rounds = int(rounds_match.group(1)) if rounds_match else 0
        if rounds_match:
            print(f"Parsed Rounds: {rounds}")
        else:
            print("Failed to parse rounds from report")
        
        return {
            "accuracy": accuracy,
            "rounds": rounds,
            "filename": os.path.basename(latest_report)
        }
    except Exception as e:
        return {"error": f"Failed to parse report: {e}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
