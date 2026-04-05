"""
ML Training Pipeline with MLflow Tracking and Governance Integration.
"""
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from src.data.lineage import lineage_tracker
from src.database.models import ModelRegistry
from src.governance.bias_detection import bias_detector
from src.database.connection import get_db_session
from config import settings


class ComplianceMLPipeline:
    """
    ML Training Pipeline with embedded governance.
    Every model is tracked, validated, and audited.
    """
    
    def __init__(self, experiment_name: str = None):
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.model_id = f"model_{uuid.uuid4().hex[:12]}"
        
        # Setup MLflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"Initialized pipeline with model_id: {self.model_id}")
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'approved',
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, str]:
        """
        Prepare data with lineage tracking and bias detection.
        
        Returns:
            X_train, X_test, y_train, y_test, dataset_id
        """
        logger.info(f"Preparing data: {len(df)} samples")
        
        # Register raw dataset
        dataset_id = lineage_tracker.register_dataset(
            dataset_name=f"social_scoring_raw_{datetime.now().strftime('%Y%m%d')}",
            source_type="synthetic",
            source_location="data/synthetic/social_scoring_latest.csv",
            df=df,
            created_by="training_pipeline"
        )
        logger.info("Running pre-training bias detection...")
        bias_report = bias_detector.detect_pre_training_bias(
            df=df,
            target_column=target_column,
            dataset_id=dataset_id
        )
        
        if bias_report['bias_detected']:
            logger.warning(f"⚠️  Bias detected (severity: {bias_report['severity']})")
            logger.warning(f"Recommendations: {bias_report['recommendations']}")
        else:
            logger.info("✓ No significant bias detected in dataset")
        
       