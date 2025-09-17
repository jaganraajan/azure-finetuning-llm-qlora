"""
Azure ML utilities for workspace management and compute cluster operations.
"""
import datetime

from typing import Optional, Dict, Any
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute, Environment
from azure.ai.ml import command
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)

class AzureMLManager:
    """Manages Azure ML workspace and compute resources."""
    
    def __init__(self):
        """Initialize Azure ML manager."""
        self.config = get_config()
        self.credential = DefaultAzureCredential()
        self.ml_client = self._create_ml_client()
    
    def _create_ml_client(self) -> MLClient:
        """Create Azure ML client."""
        try:
            ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.config.get("azure.subscription_id"),
                resource_group_name=self.config.get("azure.resource_group"),
                workspace_name=self.config.get("azure.ml_workspace"),
            )
            logger.info("Successfully connected to Azure ML workspace")
            return ml_client
        except Exception as e:
            logger.error(f"Failed to connect to Azure ML workspace: {e}")
            raise
    
    def create_compute_cluster(self, force_recreate: bool = False) -> AmlCompute:
        """Create or get existing compute cluster."""
        cluster_name = self.config.get("compute.cluster_name")
        
        try:
            # Check if cluster already exists
            if not force_recreate:
                compute = self.ml_client.compute.get(cluster_name)
                logger.info(f"Using existing compute cluster: {cluster_name}")
                return compute
        except ResourceNotFoundError:
            pass
        
        # Create new compute cluster
        logger.info(f"Creating compute cluster: {cluster_name}")
        
        compute_config = AmlCompute(
            name=cluster_name,
            type="amlcompute",
            size=self.config.get("compute.vm_size"),
            min_instances=self.config.get("compute.min_nodes"),
            max_instances=self.config.get("compute.max_nodes"),
            idle_time_before_scale_down=self.config.get("compute.idle_seconds_before_scaledown", 1800),
        )
        
        try:
            compute = self.ml_client.compute.begin_create_or_update(compute_config).result()
            logger.info(f"Successfully created compute cluster: {cluster_name}")
            return compute
        except Exception as e:
            logger.error(f"Failed to create compute cluster: {e}")
            raise
    
    def create_environment(self) -> Environment:
        """Create custom environment for training."""
        env_name = "healthcare-qlora-env"
        
        try:
            # Check if environment already exists
            environment = self.ml_client.environments.get(env_name, version="1")
            logger.info(f"Using existing environment: {env_name}")
            return environment
        except ResourceNotFoundError:
            pass
        
        # Create new environment
        logger.info(f"Creating environment: {env_name}")
        
        environment = Environment(
            name=env_name,
            description="Environment for healthcare QLoRA fine-tuning",
            conda_file="config/conda_env.yaml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )
        
        try:
            environment = self.ml_client.environments.create_or_update(environment)
            logger.info(f"Successfully created environment: {env_name}")
            return environment
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            raise
    
    def submit_training_job(
        self, 
        training_script: str,
        experiment_name: str = "healthcare-qlora-training",
        **kwargs
    ):
        """Submit training job to Azure ML using command() API."""
        
        logger.info(f"Submitting training job: {experiment_name}")
        
        # Create compute cluster if needed
        compute = self.create_compute_cluster()
        
        # Create environment
        environment = self.create_environment()
        
        # Generate a unique job name
        unique_job_name = f"qlora-training-job-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    
        # Create command job using command() API
        job = command(
            code="./src",
            command=f"python {training_script}",
            environment=environment,
            compute=compute.name,
            experiment_name=experiment_name,
            name=unique_job_name,
            description="Healthcare QLoRA fine-tuning job",
            tags={"project": "healthcare-qlora", "type": "training"},
            **kwargs
        )
        
        try:
            job = self.ml_client.jobs.create_or_update(job)
            logger.info(f"Successfully submitted job: {job.name}")
            return job
        except Exception as e:
            logger.error(f"Failed to submit training job: {e}")
            raise
    
    def register_model(
        self, 
        model_path: str, 
        model_name: str,
        model_version: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Register trained model in Azure ML."""
        
        logger.info(f"Registering model: {model_name}")
        
        model_tags = tags or {}
        model_tags.update({
            "framework": "transformers",
            "technique": "qlora",
            "domain": "healthcare"
        })
        
        from azure.ai.ml.entities import Model
        
        model = Model(
            name=model_name,
            version=model_version,
            path=model_path,
            description=description or "Healthcare QLoRA fine-tuned model",
            tags=model_tags,
        )
        
        try:
            registered_model = self.ml_client.models.create_or_update(model)
            logger.info(f"Successfully registered model: {registered_model.name}:{registered_model.version}")
            return registered_model
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """Get status of a training job."""
        try:
            job = self.ml_client.jobs.get(job_name)
            return {
                "name": job.name,
                "status": job.status,
                "creation_time": job.creation_context.created_at,
                "experiment_name": job.experiment_name,
                "compute": job.compute,
            }
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise