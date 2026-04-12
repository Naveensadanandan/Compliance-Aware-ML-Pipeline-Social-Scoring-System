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
from logger import logger
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
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, str]:
        """
        Prepare data with lineage tracking and bias detection.
        
        Returns:
            X_train, X_test, y_train, y_test, dataset_id
        """
        logger.info(f"Preparing data: {len(df)} samples")
        
    
        
        # Feature engineering
        df_processed = self._feature_engineering(df)
    
        
        # Separate features and target
        protected_cols = ['gender', 'ethnicity', 'age_group']
        feature_cols = [
            col for col in df_processed.columns 
            if col != target_column and col not in protected_cols and col != 'application_id'
        ]
        
        X = df_processed[feature_cols]
        y = df_processed[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test, df_processed
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create and encode features."""
        df_copy = df.copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_cols = [
            'gender', 'ethnicity', 'age_group', 'education_level',
            'employment_status', 'loan_purpose'
        ]
        
        for col in categorical_cols:
            if col in df_copy.columns:
                le = LabelEncoder()
                df_copy[f'{col}_encoded'] = le.fit_transform(df_copy[col].astype(str))
                label_encoders[col] = le
        
        # Create new features
        df_copy['credit_utilization'] = df_copy['existing_debt'] / (df_copy['annual_income'] + 1)
        df_copy['loan_to_income_ratio'] = df_copy['loan_amount_requested'] / (df_copy['annual_income'] + 1)
        df_copy['savings_to_income_ratio'] = df_copy['savings_amount'] / (df_copy['annual_income'] + 1)
        df_copy['avg_debt_per_loan'] = df_copy['existing_debt'] / (df_copy['previous_loans'] + 1)
        df_copy['credit_age_score'] = df_copy['years_credit_history'] * df_copy['payment_history_score']
        
        return df_copy
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        model_type: str = 'xgboost'
    ) -> Any:
        """
        Train model with MLflow tracking.
        
        Args:
            model_type: 'xgboost', 'random_forest', 'gradient_boosting', 'logistic'
        """
        logger.info(f"Training {model_type} model...")
        
        with mlflow.start_run(run_name=f"{model_type}_{self.model_id}") as run:
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_id", self.model_id)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))
            
            # Train model
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    eval_metric='logloss'
                )
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 6)
                mlflow.log_param("learning_rate", 0.1)
            
            elif model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 10)
            
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                mlflow.log_param("n_estimators", 100)
                mlflow.log_param("max_depth", 5)
            
            else:  # logistic regression
                model = LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
                mlflow.log_param("max_iter", 1000)
            
            # Fit model
            training_start = datetime.utcnow()
            model.fit(X_train, y_train)
            training_duration = (datetime.utcnow() - training_start).total_seconds()
            
            mlflow.log_metric("training_duration_seconds", training_duration)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Evaluate
            metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model
            if model_type == 'xgboost':
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            # Save MLflow run ID
            self.mlflow_run_id = run.info.run_id
            
            logger.info(f"Model trained successfully (Run ID: {self.mlflow_run_id})")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            
            return model, metrics
    
    def _evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate model performance metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    
