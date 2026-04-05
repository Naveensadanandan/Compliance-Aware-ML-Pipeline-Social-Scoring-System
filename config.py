"""
Configuration management for the compliance-aware ML pipeline.
Loads settings from environment variables with validation.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Database
    database_url: str = Field(
        default="postgresql://postgres:password@localhost:5432/compliance_ml",
        description="PostgreSQL connection string"
    )
    database_host: str = "localhost"
    database_port: int = 5432
    database_name: str = "compliance_ml"
    database_user: str = "postgres"
    database_password: str = "password"
    
    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "social_scoring_compliance"
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Governance Flags
    enable_bias_detection: bool = True
    enable_explainability: bool = True
    enable_audit_logging: bool = True
    enable_data_lineage: bool = True
    
    # Bias Thresholds
    bias_threshold_statistical_parity: float = 0.1
    bias_threshold_equal_opportunity: float = 0.1
    bias_threshold_predictive_parity: float = 0.1
    
    # Protected Attributes
    protected_attributes: str = "gender,ethnicity,age_group"
    
    @property
    def protected_attributes_list(self) -> List[str]:
        """Parse protected attributes into a list."""
        return [attr.strip() for attr in self.protected_attributes.split(",")]
    
    # Model Configuration
    model_registry_path: str = "./models"
    model_version: str = "production"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/application.log"
    
    # Monitoring
    prometheus_port: int = 9090
    enable_monitoring: bool = True
    
    # Data Paths
    data_raw_path: str = "./data/raw"
    data_processed_path: str = "./data/processed"
    data_synthetic_path: str = "./data/synthetic"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        paths = [
            self.model_registry_path,
            self.data_raw_path,
            self.data_processed_path,
            self.data_synthetic_path,
            os.path.dirname(self.log_file),
        ]
        
        for path in paths:
            Path(path).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()
