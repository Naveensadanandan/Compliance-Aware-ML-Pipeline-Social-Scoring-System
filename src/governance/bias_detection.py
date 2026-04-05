"""
Bias Detection for ML Governance.
Pre-training and post-training fairness analysis.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
from sklearn.metrics import confusion_matrix

from src.database.models import BiasDetectionReport
from src.database.connection import get_db_session
from config import settings


class BiasDetector:
    """
    Comprehensive bias detection for ML fairness.
    Implements multiple fairness metrics aligned with EU AI Act requirements.
    """
    
    def __init__(self):
        self.protected_attributes = settings.protected_attributes_list
        self.thresholds = {
            'statistical_parity': settings.bias_threshold_statistical_parity,
            'equal_opportunity': settings.bias_threshold_equal_opportunity,
            'predictive_parity': settings.bias_threshold_predictive_parity
        }
    
    def detect_pre_training_bias(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_id: str,
        protected_attrs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze dataset for bias before model training.
        Focuses on data representation and label distribution.
        
        Args:
            df: Input dataset
            target_column: Name of target variable
            dataset_id: Lineage dataset ID
            protected_attrs: List of protected attributes to analyze
            
        Returns:
            Bias report with metrics and recommendations
        """
        if protected_attrs is None:
            protected_attrs = [attr for attr in self.protected_attributes if attr in df.columns]
        
        logger.info(f"Running pre-training bias detection on {len(df)} records")
        logger.info(f"Protected attributes: {protected_attrs}")
        
        report = {
            'report_id': f"bias_pre_{uuid.uuid4().hex[:12]}",
            'detection_type': 'pre-training',
            'dataset_id': dataset_id,
            'timestamp': datetime.utcnow().isoformat(),
            'protected_attributes': protected_attrs,
            'metrics': {},
            'bias_detected': False,
            'severity': 'none',
            'recommendations': []
        }
        
        # Analyze each protected attribute
        for attr in protected_attrs:
            if attr not in df.columns:
                logger.warning(f"Protected attribute '{attr}' not found in dataset")
                continue
            
            attr_metrics = self._analyze_attribute_bias(df, attr, target_column)
            report['metrics'][attr] = attr_metrics
            
            # Check for bias
            if attr_metrics['bias_flags']:
                report['bias_detected'] = True
        
        # Determine overall severity
        report['severity'] = self._calculate_severity(report['metrics'])
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['metrics'])
        
        # Save to database
        self._save_bias_report(report)
        
        return report
    
    def _analyze_attribute_bias(
        self,
        df: pd.DataFrame,
        protected_attr: str,
        target_column: str
    ) -> Dict[str, Any]:
        """Analyze bias for a specific protected attribute."""
        metrics = {
            'attribute': protected_attr,
            'unique_values': df[protected_attr].nunique(),
            'distribution': {},
            'target_distribution': {},
            'statistical_parity_difference': {},
            'bias_flags': []
        }
        
        # Overall distribution
        value_counts = df[protected_attr].value_counts()
        total = len(df)
        metrics['distribution'] = {
            str(k): {'count': int(v), 'percentage': float(v/total)}
            for k, v in value_counts.items()
        }
        
        # Target distribution per group
        target_overall_rate = df[target_column].mean()
        
        for group in df[protected_attr].unique():
            group_data = df[df[protected_attr] == group]
            group_target_rate = group_data[target_column].mean()
            group_size = len(group_data)
            
            # Statistical parity difference
            spd = group_target_rate - target_overall_rate
            
            metrics['target_distribution'][str(group)] = {
                'group_size': int(group_size),
                'positive_rate': float(group_target_rate),
                'statistical_parity_difference': float(spd)
            }
            
            # Flag if difference exceeds threshold
            if abs(spd) > self.thresholds['statistical_parity']:
                metrics['bias_flags'].append({
                    'group': str(group),
                    'metric': 'statistical_parity_difference',
                    'value': float(spd),
                    'threshold': self.thresholds['statistical_parity'],
                    'message': f"Group '{group}' has {spd:.2%} difference in positive outcomes"
                })
        
        # Representation bias (Imbalanced groups)
        min_group_size = min(v['count'] for v in metrics['distribution'].values())
        max_group_size = max(v['count'] for v in metrics['distribution'].values())
        
        if max_group_size > 10 * min_group_size:  # 10x imbalance
            metrics['bias_flags'].append({
                'metric': 'representation_imbalance',
                'ratio': float(max_group_size / min_group_size),
                'message': f"Severe representation imbalance: largest group is {max_group_size/min_group_size:.1f}x larger"
            })
        
        return metrics
    
    def detect_post_training_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_data: pd.DataFrame,
        model_id: str,
        protected_attrs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze model predictions for bias after training.
        Focuses on prediction fairness across protected groups.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            protected_data: DataFrame with protected attributes
            model_id: Model identifier
            protected_attrs: List of protected attributes
            
        Returns:
            Bias report with fairness metrics
        """
        if protected_attrs is None:
            protected_attrs = [attr for attr in self.protected_attributes if attr in protected_data.columns]
        
        logger.info(f"Running post-training bias detection on {len(y_true)} predictions")
        
        report = {
            'report_id': f"bias_post_{uuid.uuid4().hex[:12]}",
            'detection_type': 'post-training',
            'model_id': model_id,
            'timestamp': datetime.utcnow().isoformat(),
            'protected_attributes': protected_attrs,
            'metrics': {},
            'bias_detected': False,
            'severity': 'none',
            'recommendations': []
        }
        
        # Analyze each protected attribute
        for attr in protected_attrs:
            if attr not in protected_data.columns:
                continue
            
            attr_metrics = self._analyze_prediction_fairness(
                y_true, y_pred, protected_data[attr]
            )
            report['metrics'][attr] = attr_metrics
            
            if attr_metrics['bias_flags']:
                report['bias_detected'] = True
        
        # Severity and recommendations
        report['severity'] = self._calculate_severity(report['metrics'])
        report['recommendations'] = self._generate_recommendations(report['metrics'])
        
        # Save report
        self._save_bias_report(report)
        
        return report
    
    def _analyze_prediction_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: pd.Series
    ) -> Dict[str, Any]:
        """Analyze prediction fairness for a protected attribute."""
        metrics = {
            'attribute': protected_attr.name,
            'fairness_metrics': {},
            'bias_flags': []
        }
        
        # Calculate metrics per group
        for group in protected_attr.unique():
            group_mask = protected_attr == group
            group_y_true = y_true[group_mask]
            group_y_pred = y_pred[group_mask]
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                group_y_true, group_y_pred, labels=[0, 1]
            ).ravel()
            
            # Metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
            selection_rate = (tp + fp) / len(group_y_pred) if len(group_y_pred) > 0 else 0
            
            metrics['fairness_metrics'][str(group)] = {
                'true_positive_rate': float(tpr),
                'false_positive_rate': float(fpr),
                'positive_predictive_value': float(ppv),
                'selection_rate': float(selection_rate),
                'group_size': int(group_mask.sum())
            }
        
        # Calculate fairness differences
        groups = list(metrics['fairness_metrics'].keys())
        if len(groups) >= 2:
            # Equal Opportunity: TPR should be similar
            tprs = [metrics['fairness_metrics'][g]['true_positive_rate'] for g in groups]
            tpr_diff = max(tprs) - min(tprs)
            
            if tpr_diff > self.thresholds['equal_opportunity']:
                metrics['bias_flags'].append({
                    'metric': 'equal_opportunity_violation',
                    'value': float(tpr_diff),
                    'threshold': self.thresholds['equal_opportunity'],
                    'message': f"True positive rate differs by {tpr_diff:.2%} across groups"
                })
            
            # Predictive Parity: PPV should be similar
            ppvs = [metrics['fairness_metrics'][g]['positive_predictive_value'] for g in groups]
            ppv_diff = max(ppvs) - min(ppvs)
            
            if ppv_diff > self.thresholds['predictive_parity']:
                metrics['bias_flags'].append({
                    'metric': 'predictive_parity_violation',
                    'value': float(ppv_diff),
                    'threshold': self.thresholds['predictive_parity'],
                    'message': f"Positive predictive value differs by {ppv_diff:.2%} across groups"
                })
            
            # Demographic Parity: Selection rate should be similar
            selection_rates = [metrics['fairness_metrics'][g]['selection_rate'] for g in groups]
            selection_diff = max(selection_rates) - min(selection_rates)
            
            if selection_diff > self.thresholds['statistical_parity']:
                metrics['bias_flags'].append({
                    'metric': 'demographic_parity_violation',
                    'value': float(selection_diff),
                    'threshold': self.thresholds['statistical_parity'],
                    'message': f"Selection rate differs by {selection_diff:.2%} across groups"
                })
        
        return metrics
    
    def _calculate_severity(self, metrics: Dict[str, Any]) -> str:
        """Determine overall bias severity."""
        total_flags = sum(
            len(attr_metrics.get('bias_flags', [])) 
            for attr_metrics in metrics.values()
        )
        
        if total_flags == 0:
            return 'none'
        elif total_flags <= 2:
            return 'low'
        elif total_flags <= 5:
            return 'medium'
        elif total_flags <= 8:
            return 'high'
        else:
            return 'critical'
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on detected bias."""
        recommendations = []
        
        for attr, attr_metrics in metrics.items():
            for flag in attr_metrics.get('bias_flags', []):
                if flag['metric'] == 'statistical_parity_difference':
                    recommendations.append(
                        f"Consider rebalancing dataset for '{attr}' or using fairness constraints during training"
                    )
                elif flag['metric'] == 'representation_imbalance':
                    recommendations.append(
                        f"Collect more data for underrepresented groups in '{attr}'"
                    )
                elif flag['metric'] == 'equal_opportunity_violation':
                    recommendations.append(
                        f"Model has unequal performance across '{attr}' groups. Consider using fairness-aware algorithms"
                    )
                elif flag['metric'] == 'demographic_parity_violation':
                    recommendations.append(
                        f"Model shows demographic parity violation for '{attr}'. Review feature engineering"
                    )
        
        if not recommendations:
            recommendations.append("No significant bias detected. Continue monitoring.")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _save_bias_report(self, report: Dict[str, Any]):
        """Save bias report to database."""
        try:
            # Flatten metrics for database storage
            statistical_parity = {}
            equal_opportunity = {}
            disparate_impact = {}
            
            for attr, metrics in report['metrics'].items():
                if 'target_distribution' in metrics:
                    statistical_parity[attr] = metrics['target_distribution']
                if 'fairness_metrics' in metrics:
                    equal_opportunity[attr] = metrics['fairness_metrics']
            
            db_report = BiasDetectionReport(
                report_id=report['report_id'],
                detection_type=report['detection_type'],
                dataset_id=report.get('dataset_id'),
                model_id=report.get('model_id'),
                statistical_parity_difference=statistical_parity,
                equal_opportunity_difference=equal_opportunity,
                average_odds_difference={},
                disparate_impact=disparate_impact,
                bias_detected=report['bias_detected'],
                severity=report['severity'],
                recommendations='\n'.join(report['recommendations']),
                created_at=datetime.utcnow(),
                created_by='bias_detector',
                protected_attributes_analyzed=report['protected_attributes']
            )
            
            with get_db_session() as session:
                session.add(db_report)
                session.commit()
                logger.info(f"Saved bias report: {report['report_id']}")
        
        except Exception as e:
            logger.error(f"Failed to save bias report: {e}")


# Global instance
bias_detector = BiasDetector()
