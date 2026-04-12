"""
Prefect Workflow Orchestration for Compliance-Aware ML Pipeline.
Orchestrates the entire pipeline from data generation to monitoring.
"""
from prefect import flow, task
from typing import Dict, Any
from logger import logger

logger.info("This goes into the file")
import pandas as pd

# Import pipeline components
from scripts.generate_data import generate_social_scoring_data, save_dataset
from src.training.pipeline import ComplianceMLPipeline


@task(name="Generate Synthetic Data", retries=2)
def generate_data_task(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic social scoring dataset."""
    logger.info(f"Task: Generating {n_samples} samples")
    df = generate_social_scoring_data(n_samples)
    save_dataset(df)
    logger.info("✓ Data generation completed")
    return df


@task(name="Train ML Model with Governance")
def train_model_task(df: pd.DataFrame) -> Dict[str, Any]:
    """Train model with MLflow tracking."""
    logger.info("Task: Training ML model")
    
    pipeline = ComplianceMLPipeline()
    
    # Prepare data
    X_train, X_test, y_train, y_test, df_processed = pipeline.prepare_data(df)
    
    # Train model
    model, metrics = pipeline.train_model(X_train, y_train, X_test, y_test, model_type='xgboost')
    
    # Post-training validation
    protected_data = df_processed[['gender', 'ethnicity', 'age_group']].iloc[y_test.index]
    
    
    logger.info(f"✓ Model training completed: {pipeline.model_id}")
    
    return {
        'model_id': pipeline.model_id,
        'metrics': metrics
    }


@flow(
    name="Compliance-Aware ML Pipeline",
    description="End-to-end ML pipeline with embedded governance"
)
def compliance_ml_pipeline_flow(
    n_samples: int = 10000
) -> Dict[str, Any]:
    """
    Main workflow orchestrating the entire compliance-aware ML pipeline.
    
    Pipeline stages:
    1. Data Generation
    2. Lineage Tracking
    3. Pre-training Bias Detection (Governance Gate)
    4. Model Training with MLflow
    5. Post-training Validation
    6. Deployment Approval (Governance Gate)
    7. Monitoring Setup
    
    Args:
        n_samples: Number of samples to generate
        enable_governance_gates: If True, pipeline halts on governance failures
        
    Returns:
        Pipeline execution results
    """
    logger.info("=" * 80)
    logger.info("STARTING COMPLIANCE-AWARE ML PIPELINE")
    logger.info("=" * 80)
    
    # Stage 1: Data Generation
    df = generate_data_task(n_samples)
    
    
    # Stage 4: Model Training
    training_results = train_model_task(df)
    
    logger.info("=" * 80)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    
    return {
        'status': 'success',
        'model_id': training_results['model_id'],
        'metrics': training_results['metrics'],
        'governance': {
            'post_training_bias': training_results['governance']['bias_report'],
            'fairness_approved': training_results['fairness_approved']
        }
    }


if __name__ == "__main__":
    # Run the main pipeline
    result = compliance_ml_pipeline_flow(n_samples=10000)
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE RESULTS")
    logger.info("=" * 80)
    logger.info(f"Status: {result['status']}")
    logger.info(f"Model ID: {result.get('model_id')}")
    
