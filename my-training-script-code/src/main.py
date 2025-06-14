import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import time
import boto3
import logging
import atexit
import signal
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded bucket for testing
S3_BUCKET = 'mlflow-bucket-aws-batch-12june2025-sachin'
S3_PREFIX = 'models'

# MLflow configuration
MLFLOW_TRACKING_SERVER_ARN = "arn:aws:sagemaker:us-east-1:532870744519:mlflow-tracking-server/batch-mlflow-integration-server"
EXPERIMENT_NAME = "mnist-training"

# Define the neural network class
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def setup_mlflow():
    """Setup MLflow tracking"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_SERVER_ARN)
        logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
            logger.info(f"Created new experiment with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment with ID: {experiment.experiment_id}")
        
        mlflow.set_experiment(EXPERIMENT_NAME)
        return True
    except Exception as e:
        logger.error(f"Failed to setup MLflow: {str(e)}")
        return False


def save_model_to_s3(model, bucket, key):
    """Actually saves the model to S3"""
    try:
        # Save model locally first
        local_path = '/tmp/model.pth'
        torch.save(model.state_dict(), local_path)
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_path, bucket, key)
        logger.info(f"Successfully saved model to s3://{bucket}/{key}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)


def cleanup_handler(model, bucket, key):
    """Cleanup handler for exit"""
    try:
        save_model_to_s3(model, bucket, key)
    except Exception as e:
        logger.error(f"Error in cleanup: {str(e)}")


def main():
    # Print AWS Batch environment variables
    print(f"AWS_BATCH_JOB_ID: {os.environ.get('AWS_BATCH_JOB_ID', 'Not running in AWS Batch')}")
    print(f"AWS_BATCH_JOB_ATTEMPT: {os.environ.get('AWS_BATCH_JOB_ATTEMPT', 'Not running in AWS Batch')}")

    # Setup MLflow
    mlflow_enabled = setup_mlflow()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load and transform data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True
    )

    # Initialize network and move to device
    net = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Register cleanup handlers
    atexit.register(cleanup_handler, net, S3_BUCKET, f'{S3_PREFIX}/final.pth')
    signal.signal(signal.SIGTERM, lambda signo, frame: cleanup_handler(net, S3_BUCKET, f'{S3_PREFIX}/final.pth'))

    # Training loop with MLflow logging
    print("Starting training...")
    start_time = time.time()
    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params({
            "epochs": 2,
            "batch_size": 64,
            "learning_rate": 0.001,
            "model_type": "SimpleNet",
            "device": str(device)
        })

        for epoch in range(2):  # Just 2 epochs for testing
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    avg_loss = running_loss / 100
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
                    
                    # Log metrics with explicit step
                    try:
                        mlflow.log_metrics({
                            "loss": avg_loss,
                            "epoch": epoch + 1,
                        }, step=epoch * len(trainloader) + i)
                        logger.debug(f"Logged metrics at step {epoch * len(trainloader) + i}")
                    except Exception as e:
                        logger.error(f"Failed to log metrics: {str(e)}")
                    
                    running_loss = 0.0
            
            # Save checkpoint after each epoch
            save_model_to_s3(net, S3_BUCKET, f'{S3_PREFIX}/epoch_{epoch}.pth')
            
            # Log epoch metrics
            try:
                mlflow.log_metric("epoch_complete", epoch + 1)
            except Exception as e:
                logger.error(f"Failed to log epoch metric: {str(e)}")

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save final model
        final_key = f'{S3_PREFIX}/final.pth'
        save_model_to_s3(net, S3_BUCKET, final_key)
        print(f"Final model saved to s3://{S3_BUCKET}/{final_key}")

        # Log final metrics
        training_time = time.time() - start_time
        try:
            mlflow.log_metric("training_time", training_time)
            mlflow.pytorch.log_model(net, "model")
        except Exception as e:
            logger.error(f"Failed to log final metrics/model: {str(e)}")


if __name__ == "__main__":
    main()
