usage: python manage_pipeline.py [-h]
                          {setup,mlflow,run,config,experiments,deploy} ...

Manage Federated Learning Pipeline

positional arguments:
  {setup,mlflow,run,config,experiments,deploy}
                        Available commands
    setup               Setup environment
    mlflow              Start MLflow server
    run                 Run distributed FL pipeline
    config              Show current configuration
    experiments         Show MLflow experiments
    deploy              Deploy pipeline
after that, run "prefect server start" to see prefect ui
run "mlflow ui" to see mlflow ui
