"""
Database models for governance, audit logging, and data lineage.
"""
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    Text, JSON, ForeignKey, Table, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()


class DataLineage(Base):
    """Track data provenance and transformations."""
    __tablename__ = 'data_lineage'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(100), unique=True, nullable=False, index=True)
    dataset_name = Column(String(200), nullable=False)
    source_type = Column(String(50))  # 'raw', 'api', 'database', 'synthetic'
    source_location = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    
    # Lineage relationships
    parent_dataset_id = Column(String(100), ForeignKey('data_lineage.dataset_id'))
    transformation_applied = Column(Text)  # Description of transformation
    
    # Metadata
    record_count = Column(Integer)
    column_count = Column(Integer)
    schema_info = Column(JSON)  # Column names, types
    quality_metrics = Column(JSON)  # Completeness, validity, etc.
    
    # Relationships
    parent = relationship("DataLineage", remote_side=[dataset_id], backref="children")


class BiasDetectionReport(Base):
    """Store bias detection results."""
    __tablename__ = 'bias_detection_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(100), unique=True, nullable=False, index=True)
    detection_type = Column(String(50))  # 'pre-training', 'post-training'
    dataset_id = Column(String(100), ForeignKey('data_lineage.dataset_id'))
    model_id = Column(String(100), nullable=True)  # For post-training
    
    # Metrics
    statistical_parity_difference = Column(JSON)  # Per protected attribute
    equal_opportunity_difference = Column(JSON)
    average_odds_difference = Column(JSON)
    disparate_impact = Column(JSON)
    
    # Overall assessment
    bias_detected = Column(Boolean, default=False)
    severity = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    recommendations = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100))
    protected_attributes_analyzed = Column(JSON)


class ModelRegistry(Base):
    """Track trained models and their governance metadata."""
    __tablename__ = 'model_registry'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    model_name = Column(String(200), nullable=False)
    model_version = Column(String(50))
    model_type = Column(String(100))  # 'XGBoost', 'RandomForest', etc.
    
    # Training info
    training_dataset_id = Column(String(100), ForeignKey('data_lineage.dataset_id'))
    training_started_at = Column(DateTime)
    training_completed_at = Column(DateTime)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # Governance
    bias_report_id = Column(String(100), ForeignKey('bias_detection_reports.report_id'))
    fairness_approved = Column(Boolean, default=False)
    explainability_score = Column(Float)  # Average SHAP value stability
    
    # MLflow integration
    mlflow_run_id = Column(String(100))
    mlflow_experiment_id = Column(String(100))
    
    # Status
    status = Column(String(50))  # 'training', 'validation', 'approved', 'deployed', 'retired'
    deployment_approved_by = Column(String(100))
    deployment_approved_at = Column(DateTime)
    
    # Artifacts
    model_path = Column(String(500))
    metadata = Column(JSON)


class AuditLog(Base):
    """Comprehensive audit trail for all system actions."""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Action details
    action_type = Column(String(100))  # 'prediction', 'training', 'deployment', 'data_access'
    actor = Column(String(100))  # User or system component
    resource_type = Column(String(100))  # 'model', 'dataset', 'api_endpoint'
    resource_id = Column(String(200))
    
    # Request/Response
    request_data = Column(JSON)
    response_data = Column(JSON)
    
    # Governance
    explanation_provided = Column(Boolean, default=False)
    explanation_data = Column(JSON)  # SHAP/LIME values
    bias_checked = Column(Boolean, default=False)
    
    # Compliance
    gdpr_compliant = Column(Boolean, default=True)
    ai_act_compliant = Column(Boolean, default=True)
    
    # Result
    status = Column(String(50))  # 'success', 'failure', 'blocked'
    error_message = Column(Text, nullable=True)
    
    # Session info
    session_id = Column(String(100))
    ip_address = Column(String(45))
    user_agent = Column(String(500))


class PredictionLog(Base):
    """Detailed log of model predictions with explanations."""
    __tablename__ = 'prediction_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Model info
    model_id = Column(String(100), ForeignKey('model_registry.model_id'))
    model_version = Column(String(50))
    
    # Input features
    input_features = Column(JSON)  # Feature names and values
    
    # Prediction
    prediction = Column(Float)
    prediction_class = Column(Integer, nullable=True)  # For classification
    confidence_score = Column(Float)
    
    # Explainability
    shap_values = Column(JSON)  # Feature contributions
    lime_explanation = Column(JSON, nullable=True)
    top_features = Column(JSON)  # Top 5 contributing features
    
    # Fairness check
    protected_attributes = Column(JSON)
    fairness_flags = Column(JSON)  # Any fairness concerns
    
    # Metadata
    request_id = Column(String(100))
    user_id = Column(String(100))
    audit_log_id = Column(String(100), ForeignKey('audit_logs.log_id'))


class MonitoringMetrics(Base):
    """Store model monitoring metrics over time."""
    __tablename__ = 'monitoring_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_id = Column(String(100), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Model
    model_id = Column(String(100), ForeignKey('model_registry.model_id'))
    
    # Performance drift
    accuracy_drift = Column(Float)
    prediction_distribution = Column(JSON)
    
    # Data drift
    feature_drift_scores = Column(JSON)  # Per feature drift detection
    data_quality_score = Column(Float)
    
    # Fairness drift
    bias_metrics = Column(JSON)
    fairness_degradation = Column(Boolean, default=False)
    
    # Alerts triggered
    alerts = Column(JSON)
    
    # Period
    window_start = Column(DateTime)
    window_end = Column(DateTime)
