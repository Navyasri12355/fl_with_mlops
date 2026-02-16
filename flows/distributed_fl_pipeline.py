"""
Distributed Federated Learning Pipeline using Prefect

This pipeline orchestrates the execution of:
- 1 Flower server process (global model)
- 3 client processes (one for each machine: M01, M02, M03)
- MLflow tracking for all experiments and comparisons
"""

import os
import sys
import time
import subprocess
import signal
import json
import tempfile
import csv
import shutil
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import threading
import queue

from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.task_runners import ConcurrentTaskRunner

import mlflow
import mlflow.keras
import numpy as np


@task
def setup_mlflow_experiment(experiment_name: str) -> str:
    """Setup MLflow experiment for distributed FL tracking"""
    logger = get_run_logger()
    
    try:
        # Configure MLflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        except Exception:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        
        # Set environment variable for client processes
        os.environ["MLFLOW_EXPERIMENT"] = experiment_name
        
        return experiment_name
        
    except Exception as e:
        logger.error(f"Failed to setup MLflow experiment: {e}")
        raise


@task
def validate_distributed_setup() -> Dict[str, Any]:
    """Validate that the distributed FL setup is ready"""
    logger = get_run_logger()
    
    validation_results = {
        "status": "success",
        "errors": [],
        "warnings": [],
        "machine_data": {}
    }
    
    # Check required files
    required_files = ["client.py", "server.py", "model1.py", "individual.py"]
    for file in required_files:
        if not Path(file).exists():
            validation_results["errors"].append(f"Required file missing: {file}")
    
    # Check data directories
    train_dir = Path("new_train")
    test_dir = Path("new_test")
    
    if not train_dir.exists():
        validation_results["errors"].append("Training directory 'new_train' not found")
    
    if not test_dir.exists():
        validation_results["errors"].append("Test directory 'new_test' not found")
    
    if validation_results["errors"]:
        validation_results["status"] = "failed"
        return validation_results
    
    # Check machine data
    expected_machines = ["M01", "M02", "M03"]
    for machine in expected_machines:
        machine_dir = train_dir / machine
        machine_data = {"good": 0, "bad": 0, "issues": []}
        
        if not machine_dir.exists():
            machine_data["issues"].append(f"Machine directory {machine} not found")
            validation_results["warnings"].append(f"Machine {machine} directory missing")
        else:
            good_dir = machine_dir / "good"
            bad_dir = machine_dir / "bad"
            
            if good_dir.exists():
                machine_data["good"] = len(list(good_dir.glob("*.h5")))
            else:
                machine_data["issues"].append("Missing 'good' directory")
            
            if bad_dir.exists():
                machine_data["bad"] = len(list(bad_dir.glob("*.h5")))
            else:
                machine_data["issues"].append("Missing 'bad' directory")
        
        validation_results["machine_data"][machine] = machine_data
    
    # Check test data
    test_good = test_dir / "good"
    test_bad = test_dir / "bad"
    test_data = {"good": 0, "bad": 0}
    
    if test_good.exists():
        test_data["good"] = len(list(test_good.glob("*.h5")))
    if test_bad.exists():
        test_data["bad"] = len(list(test_bad.glob("*.h5")))
    
    validation_results["machine_data"]["test"] = test_data
    
    if validation_results["warnings"]:
        validation_results["status"] = "warning"
    
    logger.info(f"Validation completed with status: {validation_results['status']}")
    return validation_results


@task
def start_flower_server(
    num_rounds: int = 3,
    min_clients: int = 3,
    server_address: str = "127.0.0.1:8080"
) -> Dict[str, Any]:
    """Start the Flower server process"""
    logger = get_run_logger()
    
    try:
        # Prepare server command
        server_cmd = [
            sys.executable, "server.py"
        ]
        
        # Set environment variables for the server
        server_env = os.environ.copy()
        server_env["FL_NUM_ROUNDS"] = str(num_rounds)
        server_env["FL_MIN_CLIENTS"] = str(min_clients)
        server_env["FL_SERVER_ADDRESS"] = server_address
        server_env["MLFLOW_EXPERIMENT"] = os.environ.get("MLFLOW_EXPERIMENT", "distributed_fl")
        
        # Ensure MLflow tracking URI is passed
        if "MLFLOW_TRACKING_URI" not in server_env:
            server_env["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
        
        logger.info(f"Server environment: MLFLOW_EXPERIMENT={server_env['MLFLOW_EXPERIMENT']}")
        logger.info(f"Server environment: MLFLOW_TRACKING_URI={server_env['MLFLOW_TRACKING_URI']}")
        
        logger.info(f"Starting Flower server with {num_rounds} rounds, {min_clients} min clients")
        
        # Ensure log directory exists
        os.makedirs("pipeline_logs", exist_ok=True)
        server_log = open(os.path.join("pipeline_logs", "server.log"), "w")
        
        # Start server process
        server_process = subprocess.Popen(
            server_cmd,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Give server time to start
        time.sleep(5)
        
        # Check if server started successfully
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            logger.error(f"Server failed to start. STDOUT: {stdout}, STDERR: {stderr}")
            raise RuntimeError("Flower server failed to start")
        
        logger.info("Flower server started successfully")
        
        return {
            "process": server_process,
            "pid": server_process.pid,
            "address": server_address,
            "num_rounds": num_rounds,
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Failed to start Flower server: {e}")
        raise


@task
def start_flower_client(
    client_id: int,
    server_address: str = "127.0.0.1:8080",
    experiment_name: str = "distributed_fl"
) -> Dict[str, Any]:
    """Start a single Flower client process"""
    logger = get_run_logger()
    
    try:
        # Prepare client command
        client_cmd = [
            sys.executable, "client.py", str(client_id)
        ]
        
        # Set environment variables for the client
        client_env = os.environ.copy()
        client_env["MLFLOW_EXPERIMENT"] = experiment_name
        client_env["FL_SERVER_ADDRESS"] = server_address
        
        # Ensure MLflow tracking URI is passed
        if "MLFLOW_TRACKING_URI" not in client_env:
            client_env["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
        
        logger.info(f"Client M0{client_id} environment: MLFLOW_EXPERIMENT={client_env['MLFLOW_EXPERIMENT']}")
        logger.info(f"Client M0{client_id} environment: MLFLOW_TRACKING_URI={client_env['MLFLOW_TRACKING_URI']}")
        
        logger.info(f"Starting client M0{client_id}")
        
        # Ensure log directory exists and create client-specific log file
        os.makedirs("pipeline_logs", exist_ok=True)
        client_log = open(os.path.join("pipeline_logs", f"M0{client_id}.log"), "w")
        
        # Set environment variables for the client
        client_env = os.environ.copy()
        client_env["MLFLOW_EXPERIMENT"] = experiment_name
        client_env["FL_SERVER_ADDRESS"] = server_address
        client_env["FL_PIPELINE_MODE"] = "True"
        
        # Start client process
        client_process = subprocess.Popen(
            client_cmd,
            env=client_env,
            stdout=client_log,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Give client time to connect
        time.sleep(2)
        
        logger.info(f"Client M0{client_id} started successfully")
        
        return {
            "client_id": client_id,
            "machine_id": f"M0{client_id}",
            "process": client_process,
            "pid": client_process.pid,
            "status": "running"
        }
        
    except Exception as e:
        logger.error(f"Failed to start client M0{client_id}: {e}")
        raise


@task
def monitor_distributed_training(
    server_info: Dict[str, Any],
    client_infos: List[Dict[str, Any]],
    timeout_minutes: int = 60
) -> Dict[str, Any]:
    """Monitor the distributed training process"""
    logger = get_run_logger()
    
    server_process = server_info["process"]
    client_processes = [info["process"] for info in client_infos]
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    training_logs = {
        "server_output": [],
        "client_outputs": {f"M0{info['client_id']}": [] for info in client_infos},
        "status": "running",
        "completion_time": None,
        "error": None
    }
    
    try:
        # Monitor processes
        while time.time() - start_time < timeout_seconds:
            # Check server status
            if server_process.poll() is not None:
                logger.info("Flower server completed")
                break
            
            # Check if all clients finished early (unlikely but possible)
            all_clients_done = all(cp.poll() is not None for cp in client_processes)
            if all_clients_done:
                logger.info("All clients completed")
                break
            
            time.sleep(5)  # Check every 5 seconds
        
        else:
            # Timeout reached
            logger.warning(f"Training timeout reached ({timeout_minutes} minutes)")
            training_logs["status"] = "timeout"
            training_logs["error"] = f"Training exceeded {timeout_minutes} minute timeout"
        
        # Ensure all processes are terminated
        if server_process.poll() is None:
            server_process.terminate()
            time.sleep(2)
            if server_process.poll() is None:
                server_process.kill()
        
        for client_process in client_processes:
            if client_process.poll() is None:
                client_process.terminate()
                time.sleep(1)
                if client_process.poll() is None:
                    client_process.kill()
        
        if training_logs["status"] == "running":
            training_logs["status"] = "completed"
            training_logs["completion_time"] = datetime.now().isoformat()
        
        logger.info(f"Distributed training monitoring completed with status: {training_logs['status']}")
        return training_logs
        
    except Exception as e:
        logger.error(f"Error during training monitoring: {e}")
        training_logs["status"] = "error"
        training_logs["error"] = str(e)
        return training_logs


@task
def collect_mlflow_results(experiment_name: str) -> Dict[str, Any]:
    """Collect and analyze results from MLflow"""
    logger = get_run_logger()
    
    try:
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        # Get all runs from this experiment
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        if runs.empty:
            logger.warning("No MLflow runs found for this experiment")
            return {"status": "no_data", "runs": [], "analysis": {}}
        
        # Organize runs by client and type
        client_runs = {"M01": [], "M02": [], "M03": []}
        eval_runs = {"M01": [], "M02": [], "M03": []}
        
        for _, run in runs.iterrows():
            run_name = run.get("tags.mlflow.runName", "")
            
            if "round" in run_name and "eval" not in run_name:
                # Training round run
                for client in client_runs.keys():
                    if client in run_name:
                        client_runs[client].append({
                            "run_id": run["run_id"],
                            "run_name": run_name,
                            "global_accuracy": run.get("metrics.global_accuracy", 0),
                            "local_accuracy": run.get("metrics.local_accuracy", 0),
                            "local_better": run.get("metrics.local_better", 0),
                            "round": run.get("params.round", 0)
                        })
            
            elif "eval" in run_name:
                # Evaluation run
                for client in eval_runs.keys():
                    if client in run_name:
                        eval_runs[client].append({
                            "run_id": run["run_id"],
                            "run_name": run_name,
                            "global_eval_accuracy": run.get("metrics.global_eval_accuracy", 0),
                            "eval_round": run.get("params.eval_round", 0)
                        })
        
        # Analyze results
        analysis = {
            "total_runs": len(runs),
            "clients_participated": len([c for c in client_runs.keys() if client_runs[c]]),
            "client_performance": {},
            "overall_summary": {}
        }
        
        # Analyze each client's performance
        for client in client_runs.keys():
            if not client_runs[client]:
                continue
            
            client_data = client_runs[client]
            
            # Sort by round
            client_data.sort(key=lambda x: int(x.get("round", 0)))
            
            # Calculate statistics
            global_accs = [r["global_accuracy"] for r in client_data if r["global_accuracy"] > 0]
            local_accs = [r["local_accuracy"] for r in client_data if r["local_accuracy"] > 0]
            local_better_count = sum(r["local_better"] for r in client_data)
            
            analysis["client_performance"][client] = {
                "rounds_completed": len(client_data),
                "final_global_accuracy": global_accs[-1] if global_accs else 0,
                "final_local_accuracy": local_accs[-1] if local_accs else 0,
                "avg_global_accuracy": np.mean(global_accs) if global_accs else 0,
                "avg_local_accuracy": np.mean(local_accs) if local_accs else 0,
                "local_better_rounds": local_better_count,
                "local_better_percentage": (local_better_count / len(client_data) * 100) if client_data else 0,
                "accuracy_improvement": (local_accs[-1] - global_accs[-1]) if (local_accs and global_accs) else 0
            }
        
        # Overall summary
        all_final_global = [analysis["client_performance"][c]["final_global_accuracy"] 
                           for c in analysis["client_performance"].keys()]
        all_final_local = [analysis["client_performance"][c]["final_local_accuracy"] 
                          for c in analysis["client_performance"].keys()]
        
        analysis["overall_summary"] = {
            "avg_final_global_accuracy": np.mean(all_final_global) if all_final_global else 0,
            "avg_final_local_accuracy": np.mean(all_final_local) if all_final_local else 0,
            "clients_prefer_local": len([c for c in analysis["client_performance"].keys() 
                                       if analysis["client_performance"][c]["local_better_percentage"] > 50]),
            "best_performing_client": max(analysis["client_performance"].keys(), 
                                        key=lambda c: analysis["client_performance"][c]["final_local_accuracy"]) 
                                        if analysis["client_performance"] else None
        }
        
        logger.info(f"Collected results from {len(runs)} MLflow runs")
        
        return {
            "status": "success",
            "runs": runs.to_dict('records'),
            "client_runs": client_runs,
            "eval_runs": eval_runs,
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Failed to collect MLflow results: {e}")
        return {"status": "error", "error": str(e), "runs": [], "analysis": {}}


@task
def select_and_save_best_model(mlflow_results: Dict[str, Any], experiment_name: str):
    """Select the best performing model and save it to best_model/ directory"""
    logger = get_run_logger()
    
    if mlflow_results["status"] != "success":
        logger.warning("MLflow results not available for best model selection")
        return
        
    analysis = mlflow_results["analysis"]
    client_performance = analysis.get("client_performance", {})
    
    best_acc = -1.0
    best_model_name = "none"
    best_run_id = None
    
    # Check global model (using avg global accuracy as representative)
    global_acc = analysis["overall_summary"]["avg_final_global_accuracy"]
    if global_acc > best_acc:
        best_acc = global_acc
        best_model_name = "global_model"
        
    # Check each client local performance
    for client, perf in client_performance.items():
        if perf["final_local_accuracy"] > best_acc:
            best_acc = perf["final_local_accuracy"]
            best_model_name = f"local_model_{client}"
            # Find the run ID for the latest round for this client
            # The client_runs dict in mlflow_results contains IDs
            # But simpler is to find it via client search
            
    logger.info(f"Best model identified: {best_model_name} with accuracy {best_acc:.4f}")
    
    # Setup directory
    Path("best_model").mkdir(exist_ok=True)
    
    if best_model_name == "global_model":
        src = "final_model_weights.weights.h5"
        if os.path.exists(src):
            shutil.copy(src, os.path.join("best_model", "best_model.weights.h5"))
            logger.info("Saved global model as best_model.weights.h5")
    else:
        # It's a local model. We need to find its artifact in MLflow.
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(experiment_name)
        machine_id = best_model_name.replace("local_model_", "")
        
        query = f"params.client = '{machine_id}'"
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=query,
            order_by=["start_time DESC"]
        )
        
        # Filter for runs that have 'round' parameter in Python
        valid_runs = [r for r in runs if "round" in r.data.params]
        
        if valid_runs:
            run_id = valid_runs[0].info.run_id
            artifacts = client.list_artifacts(run_id)
            # Find the .weights.h5 file
            weights_art = next((a.path for a in artifacts if a.path.endswith(".weights.h5")), None)
            if weights_art:
                local_path = client.download_artifacts(run_id, weights_art, "best_model")
                # Rename to best_model.weights.h5
                target = os.path.join("best_model", "best_model.weights.h5")
                if os.path.exists(target):
                    os.remove(target)
                os.rename(local_path, target)
                logger.info(f"Saved local model from client {machine_id} as best_model.weights.h5")


@task
def generate_distributed_fl_report(
    mlflow_results: Dict[str, Any],
    experiment_name: str,
    training_logs: Dict[str, Any]
) -> str:
    """Generate comprehensive report for distributed FL training"""
    logger = get_run_logger()
    
    if mlflow_results["status"] != "success":
        logger.warning("MLflow results not available for report generation")
        return ""
    
    analysis = mlflow_results["analysis"]
    
    # Create comparison table for Prefect UI
    table_data = []
    for client, perf in analysis["client_performance"].items():
        table_data.append({
            "Client": client,
            "Final Global Acc": f"{perf['final_global_accuracy']:.4f}",
            "Final Local Acc": f"{perf['final_local_accuracy']:.4f}",
            "Improvement": f"{perf['accuracy_improvement']:+.4f}",
            "Local Better %": f"{perf['local_better_percentage']:.1f}%",
            "Rounds": perf['rounds_completed']
        })
    
    create_table_artifact(
        key="distributed-fl-results",
        table=table_data,
        description="Distributed Federated Learning Results by Client"
    )
    
    # Create detailed markdown report
    report_content = f"""# Distributed Federated Learning Report

**Experiment:** {experiment_name}  
**Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Training Status:** {training_logs.get('status', 'unknown')}

## Overall Summary

- **Average Final Global Accuracy:** {analysis['overall_summary']['avg_final_global_accuracy']:.4f}
- **Average Final Local Accuracy:** {analysis['overall_summary']['avg_final_local_accuracy']:.4f}
- **Clients Preferring Local Models:** {analysis['overall_summary']['clients_prefer_local']}/3
- **Best Performing Client:** {analysis['overall_summary']['best_performing_client']}

## Client Performance Details

"""
    
    for client, perf in analysis["client_performance"].items():
        report_content += f"""### {client}

- **Rounds Completed:** {perf['rounds_completed']}
- **Final Global Accuracy:** {perf['final_global_accuracy']:.4f}
- **Final Local Accuracy:** {perf['final_local_accuracy']:.4f}
- **Accuracy Improvement:** {perf['accuracy_improvement']:+.4f}
- **Local Model Better:** {perf['local_better_rounds']}/{perf['rounds_completed']} rounds ({perf['local_better_percentage']:.1f}%)

"""
    
    # Add recommendations
    report_content += """## Recommendations

"""
    
    overall = analysis['overall_summary']
    if overall['avg_final_local_accuracy'] > overall['avg_final_global_accuracy'] + 0.02:
        report_content += "- **Local models significantly outperform global model.** Consider personalized approaches or investigate data heterogeneity.\n"
    elif overall['avg_final_global_accuracy'] > overall['avg_final_local_accuracy'] + 0.02:
        report_content += "- **Global federated model shows clear benefits.** Federated learning is working well for this dataset.\n"
    else:
        report_content += "- **Performance is similar between local and global models.** Consider privacy and infrastructure trade-offs.\n"
    
    if overall['clients_prefer_local'] >= 2:
        report_content += "- **Majority of clients perform better with local models.** Investigate data distribution and consider hybrid approaches.\n"
    
    # Add training logs summary if available
    if training_logs.get('status') == 'completed':
        report_content += "\n## Training Execution\n\n- Training completed successfully\n"
    elif training_logs.get('status') == 'timeout':
        report_content += "\n## Training Execution\n\n- Training exceeded timeout limit\n"
    elif training_logs.get('status') == 'error':
        report_content += f"\n## Training Execution\n\n- Training failed: {training_logs.get('error', 'Unknown error')}\n"
    
    # Create Prefect artifact
    create_markdown_artifact(
        key="distributed-fl-report",
        markdown=report_content,
        description="Comprehensive Distributed Federated Learning Report"
    )
    
    # Save report to file
    Path("reports").mkdir(exist_ok=True)
    report_filename = f"distributed_fl_report_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = os.path.join("reports", report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Log report as MLflow artifact
    try:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="distributed_fl_summary"):
            mlflow.log_artifact(report_path, "reports")
            
            # Log summary metrics
            mlflow.log_metrics({
                "avg_final_global_accuracy": overall['avg_final_global_accuracy'],
                "avg_final_local_accuracy": overall['avg_final_local_accuracy'],
                "clients_prefer_local": overall['clients_prefer_local'],
                "total_clients": len(analysis["client_performance"])
            })
            
            # Log per-machine metrics across all rounds for graphing
            client_runs = mlflow_results.get("client_runs", {})
            for mid, runs_info in client_runs.items():
                for run_data in runs_info:
                    round_num_str = run_data.get("round", "0")
                    try:
                        round_num = int(round_num_str)
                    except (ValueError, TypeError):
                        round_num = 0
                    
                    local_acc = float(run_data.get("local_accuracy", 0))
                    global_acc = float(run_data.get("global_accuracy", 0))
                    
                    if round_num > 0:
                        mlflow.log_metric(f"{mid}_local_accuracy", local_acc, step=round_num)
                        mlflow.log_metric(f"{mid}_global_accuracy", global_acc, step=round_num)

    except Exception as e:
        logger.warning(f"Failed to log report to MLflow: {e}")
    
    logger.info(f"Distributed FL report generated: {report_path}")
    return report_path


@flow(
    name="Distributed Federated Learning Pipeline",
    task_runner=ConcurrentTaskRunner(),
    description="Orchestrates distributed FL with 1 server + 3 clients, full MLflow tracking"
)
def distributed_fl_pipeline(
    experiment_name: str = "distributed_fl_experiment",
    num_rounds: int = 3,
    timeout_minutes: int = 60,
    server_address: str = "127.0.0.1:8080"
) -> Dict[str, Any]:
    """
    Complete distributed federated learning pipeline
    
    Args:
        experiment_name: MLflow experiment name
        num_rounds: Number of federated learning rounds
        timeout_minutes: Maximum time to wait for training completion
        server_address: Flower server address
    
    Returns:
        Complete pipeline results with analysis
    """
    logger = get_run_logger()
    
    logger.info(f"Starting distributed FL pipeline: {experiment_name}")
    logger.info(f"Configuration: {num_rounds} rounds, {timeout_minutes}min timeout")
    
    # Phase 1: Setup and Validation
    logger.info("Phase 1: Setup and validation")
    validation_results = validate_distributed_setup()
    
    if validation_results["status"] == "failed":
        logger.error("Setup validation failed")
        raise ValueError(f"Validation errors: {validation_results['errors']}")
    
    experiment_name = setup_mlflow_experiment(experiment_name)
    
    # Phase 2: Start Distributed Training
    logger.info("Phase 2: Starting distributed training processes")
    
    # Start server
    server_info = start_flower_server(
        num_rounds=num_rounds,
        server_address=server_address
    )
    
    # Start clients concurrently
    client_futures = []
    for client_id in [1, 2, 3]:  # M01, M02, M03
        future = start_flower_client.submit(
            client_id=client_id,
            server_address=server_address,
            experiment_name=experiment_name
        )
        client_futures.append(future)
    
    # Collect client info
    client_infos = []
    for future in client_futures:
        client_info = future.result()
        client_infos.append(client_info)
    
    # Phase 3: Monitor Training
    logger.info("Phase 3: Monitoring distributed training")
    training_logs = monitor_distributed_training(
        server_info=server_info,
        client_infos=client_infos,
        timeout_minutes=timeout_minutes
    )
    
    # Phase 4: Collect and Analyze Results
    logger.info("Phase 4: Collecting and analyzing results")
    
    # Wait a bit for MLflow to sync
    time.sleep(5)
    
    mlflow_results = collect_mlflow_results(experiment_name)
    
    # Phase 5: Generate Report
    logger.info("Phase 5: Generating comprehensive report")
    report_path = generate_distributed_fl_report(
        mlflow_results=mlflow_results,
        experiment_name=experiment_name,
        training_logs=training_logs
    )
    
    # Phase 6: Best Model Selection
    logger.info("Phase 6: Selecting and saving best model")
    select_and_save_best_model(mlflow_results, experiment_name)
    
    # Compile final results
    final_results = {
        "experiment_name": experiment_name,
        "validation_results": validation_results,
        "server_info": {k: v for k, v in server_info.items() if k != "process"},
        "client_infos": [{k: v for k, v in info.items() if k != "process"} for info in client_infos],
        "training_logs": training_logs,
        "mlflow_results": mlflow_results,
        "report_path": report_path,
        "configuration": {
            "num_rounds": num_rounds,
            "timeout_minutes": timeout_minutes,
            "server_address": server_address
        },
        "status": "completed" if training_logs["status"] == "completed" else "failed"
    }
    
    logger.info(f"Distributed FL pipeline completed with status: {final_results['status']}")
    
    return final_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="test_distributed_fl")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--timeout_minutes", type=int, default=60)
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    # Pass args to the pipeline
    result = distributed_fl_pipeline(
        experiment_name=args.experiment_name,
        num_rounds=args.num_rounds,
        timeout_minutes=args.timeout_minutes,
        server_address=args.server_address
    )
    
    print(f"Pipeline Status: {result['status']}")
    if result.get('mlflow_results', {}).get('status') == 'success':
        analysis = result['mlflow_results']['analysis']
        if 'overall_summary' in analysis:
            print(f"Average Global Accuracy: {analysis['overall_summary']['avg_final_global_accuracy']:.4f}")
            print(f"Average Local Accuracy: {analysis['overall_summary']['avg_final_local_accuracy']:.4f}")
            print(f"Clients Preferring Local: {analysis['overall_summary']['clients_prefer_local']}/3")