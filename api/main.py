from src.training.pipeline import ComplianceMLPipeline
import pandas as pd

if __name__ == "__main__":
    # Initialize the training pipeline
    pipeline = ComplianceMLPipeline(experiment_name="social_scoring_model_training")
    
    # Load synthetic dataset
    df = pd.read_csv("data/synthetic/social_scoring_latest.csv")
    
    # Prepare data and register lineage
    dataset_id = pipeline.prepare_data(df=df)
    
    # Further steps: train model, evaluate, log metrics, etc.