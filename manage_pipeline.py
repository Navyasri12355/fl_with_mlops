#!/usr/bin/env python3
"""
Pipeline management utility for Prefect Federated Learning project

This script provides convenient commands for managing the pipeline:
- Setup and configuration
- Running different pipeline variants
- Monitoring and analysis
- Deployment management
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import yaml
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """Setup the environment for running pipelines"""
    print("🔧 Setting up environment...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: No virtual environment detected. Consider using a virtual environment.")
    
    # Check required packages
    try:
        import prefect
        import mlflow
        import tensorflow
        print(f"✅ Prefect version: {prefect.__version__}")
        print(f"✅ MLflow version: {mlflow.__version__}")
        print(f"✅ TensorFlow version: {tensorflow.__version__}")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Check data directories
    train_dir = Path("new_train")
    test_dir = Path("new_test")
    
    if not train_dir.exists():
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        return False
    
    # Check for machine data
    machines = [d for d in train_dir.iterdir() if d.is_dir()]
    if not machines:
        print("❌ No machine directories found in training data")
        return False
    
    print(f"✅ Found {len(machines)} machines: {[m.name for m in machines]}")
    
    # Check MLflow setup
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        print(f"✅ MLflow tracking URI: {mlflow_uri}")
    else:
        print("💡 MLflow tracking URI not set. Using local SQLite database.")
    
    print("✅ Environment setup complete!")
    return True


def start_mlflow_server():
    """Start MLflow tracking server"""
    print("🚀 Starting MLflow server...")
    
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--host", "0.0.0.0",
        "--port", "5000"
    ]
    
    try:
        print("Starting MLflow server on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 MLflow server stopped")


def run_distributed_pipeline(**kwargs):
    """Run the distributed FL pipeline"""
    print("🚀 Running distributed FL pipeline...")
    
    try:
        from flows.distributed_fl_pipeline import distributed_fl_pipeline
        
        result = distributed_fl_pipeline(
            experiment_name=kwargs.get("experiment_name", "distributed_fl"),
            num_rounds=kwargs.get("num_rounds", 3),
            timeout_minutes=kwargs.get("timeout_minutes", 30)
        )
        
        print(f"✅ Pipeline Status: {result['status']}")
        
        if result['status'] == 'completed':
            mlflow_results = result['mlflow_results']
            if mlflow_results['status'] == 'success':
                analysis = mlflow_results['analysis']
                print(f"✅ Average Global Accuracy: {analysis['overall_summary']['avg_final_global_accuracy']:.4f}")
                print(f"✅ Average Local Accuracy: {analysis['overall_summary']['avg_final_local_accuracy']:.4f}")
                print(f"✅ Clients Preferring Local: {analysis['overall_summary']['clients_prefer_local']}/3")
            print(f"✅ Report saved: {result['report_path']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def show_config():
    """Display current configuration"""
    print("📋 Current Pipeline Configuration:")
    print("=" * 40)
    
    # Basic configuration
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    print(f"MLflow URI: {mlflow_uri}")
    
    # Check data directories
    train_dir = Path("new_train")
    test_dir = Path("new_test")
    
    if train_dir.exists():
        machines = [d.name for d in train_dir.iterdir() if d.is_dir()]
        print(f"Available Machines: {machines}")
    
    if test_dir.exists():
        test_good = test_dir / "good"
        test_bad = test_dir / "bad"
        good_files = len(list(test_good.glob("*.h5"))) if test_good.exists() else 0
        bad_files = len(list(test_bad.glob("*.h5"))) if test_bad.exists() else 0
        print(f"Test Data: {good_files} good, {bad_files} bad files")


def check_mlflow_experiments():
    """Check MLflow experiments"""
    print("📊 MLflow Experiments:")
    print("=" * 30)
    
    try:
        import mlflow
        
        # Set tracking URI
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        
        # List experiments
        experiments = mlflow.search_experiments()
        
        if not experiments:
            print("No experiments found.")
            return
        
        for exp in experiments:
            print(f"\n📁 {exp.name} (ID: {exp.experiment_id})")
            
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            print(f"   Runs: {len(runs)}")
            
            if not runs.empty:
                for _, run in runs.head(3).iterrows():  # Show first 3 runs
                    run_name = run.get('tags.mlflow.runName', 'Unnamed')
                    status = run['status']
                    print(f"   • {run_name} - {status}")
    
    except Exception as e:
        print(f"❌ Error checking MLflow: {e}")


def deploy_pipeline(deployment_name: str):
    """Deploy pipeline to Prefect"""
    print(f"🚀 Deploying {deployment_name} pipeline...")
    
    try:
        cmd = ["prefect", "deploy", "--name", deployment_name]
        subprocess.run(cmd, check=True)
        print(f"✅ Pipeline {deployment_name} deployed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manage Federated Learning Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup environment")
    
    # MLflow server command
    mlflow_parser = subparsers.add_parser("mlflow", help="Start MLflow server")
    
    # Run pipeline command
    run_parser = subparsers.add_parser("run", help="Run distributed FL pipeline")
    run_parser.add_argument("--experiment-name", help="MLflow experiment name")
    run_parser.add_argument("--num-rounds", type=int, default=3, help="Number of FL rounds")
    run_parser.add_argument("--timeout-minutes", type=int, default=30, help="Timeout in minutes")
    
    # Config commands
    config_parser = subparsers.add_parser("config", help="Show current configuration")
    
    # MLflow experiments command
    experiments_parser = subparsers.add_parser("experiments", help="Show MLflow experiments")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy pipeline")
    deploy_parser.add_argument("name", help="Deployment name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "setup":
            setup_environment()
        
        elif args.command == "mlflow":
            start_mlflow_server()
        
        elif args.command == "run":
            kwargs = {}
            if args.experiment_name:
                kwargs["experiment_name"] = args.experiment_name
            if args.num_rounds:
                kwargs["num_rounds"] = args.num_rounds
            if args.timeout_minutes:
                kwargs["timeout_minutes"] = args.timeout_minutes
            
            run_distributed_pipeline(**kwargs)
        
        elif args.command == "config":
            show_config()
        
        elif args.command == "experiments":
            check_mlflow_experiments()
        
        elif args.command == "deploy":
            deploy_pipeline(args.name)
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()