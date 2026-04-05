"""
Data Lineage Tracking for Compliance.
Tracks data provenance, transformations, and usage throughout the pipeline.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from sqlalchemy.orm import Session

from src.database.models import DataLineage
from src.database.connection import get_db_session


class LineageTracker:
    """
    Manages data lineage tracking for compliance and auditing.
    Every dataset transformation is recorded with full provenance.
    """
    
    def __init__(self):
        self.session_id = str(uuid.uuid4())
    
    def register_dataset(
        self,
        dataset_name: str,
        source_type: str,
        source_location: str,
        df: pd.DataFrame,
        parent_dataset_id: Optional[str] = None,
        transformation_applied: Optional[str] = None,
        created_by: str = "system"
    ) -> str:
        """
        Register a new dataset in the lineage tracking system.
        
        Args:
            dataset_name: Human-readable dataset name
            source_type: Origin type (raw, api, database, synthetic, transformation)
            source_location: Path or connection string
            df: The actual dataset
            parent_dataset_id: ID of parent dataset if this is a transformation
            transformation_applied: Description of transformation
            created_by: User or system that created this dataset
            
        Returns:
            dataset_id: Unique identifier for this dataset
        """
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"
        
        # Analyze schema
        schema_info = {
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'shape': df.shape
        }
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(df)
        
        # Create lineage record
        lineage_record = DataLineage(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source_type=source_type,
            source_location=source_location,
            created_at=datetime.utcnow(),
            created_by=created_by,
            parent_dataset_id=parent_dataset_id,
            transformation_applied=transformation_applied,
            record_count=len(df),
            column_count=len(df.columns),
            schema_info=schema_info,
            quality_metrics=quality_metrics
        )
        
        # Save to database
        with get_db_session() as session:
            session.add(lineage_record)
            session.commit()
            logger.info(f"Registered dataset: {dataset_id} ({dataset_name})")
        
        return dataset_id
    
    def _calculate_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics."""
        metrics = {
            'completeness': {},
            'uniqueness': {},
            'validity': {}
        }
        
        for col in df.columns:
            # Completeness
            missing_pct = df[col].isna().sum() / len(df)
            metrics['completeness'][col] = 1 - missing_pct
            
            # Uniqueness
            unique_pct = df[col].nunique() / len(df)
            metrics['uniqueness'][col] = unique_pct
            
            # Validity (basic checks)
            if pd.api.types.is_numeric_dtype(df[col]):
                metrics['validity'][col] = {
                    'has_negatives': bool((df[col] < 0).any()),
                    'has_zeros': bool((df[col] == 0).any()),
                    'has_nulls': bool(df[col].isna().any())
                }
        
        # Overall score
        avg_completeness = sum(metrics['completeness'].values()) / len(df.columns)
        metrics['overall_quality_score'] = avg_completeness
        
        return metrics
    
    def track_transformation(
        self,
        parent_dataset_id: str,
        new_dataset_name: str,
        transformation_description: str,
        df: pd.DataFrame,
        created_by: str = "system"
    ) -> str:
        """
        Track a data transformation with lineage.
        
        Args:
            parent_dataset_id: Source dataset ID
            new_dataset_name: Name for transformed dataset
            transformation_description: What was done
            df: Resulting dataset
            created_by: Who performed transformation
            
        Returns:
            dataset_id: New dataset ID
        """
        return self.register_dataset(
            dataset_name=new_dataset_name,
            source_type="transformation",
            source_location=f"transformed_from:{parent_dataset_id}",
            df=df,
            parent_dataset_id=parent_dataset_id,
            transformation_applied=transformation_description,
            created_by=created_by
        )
    
    def get_lineage_chain(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        Get complete lineage chain from source to current dataset.
        
        Returns:
            List of lineage records from root to leaf
        """
        chain = []
        
        with get_db_session() as session:
            current_id = dataset_id
            
            while current_id:
                record = session.query(DataLineage).filter_by(
                    dataset_id=current_id
                ).first()
                
                if not record:
                    break
                
                chain.append({
                    'dataset_id': record.dataset_id,
                    'dataset_name': record.dataset_name,
                    'source_type': record.source_type,
                    'created_at': record.created_at.isoformat(),
                    'transformation': record.transformation_applied,
                    'quality_score': record.quality_metrics.get('overall_quality_score', 0)
                })
                
                current_id = record.parent_dataset_id
        
        # Reverse to get root -> leaf order
        chain.reverse()
        return chain
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a dataset."""
        with get_db_session() as session:
            record = session.query(DataLineage).filter_by(
                dataset_id=dataset_id
            ).first()
            
            if not record:
                return None
            
            return {
                'dataset_id': record.dataset_id,
                'dataset_name': record.dataset_name,
                'source_type': record.source_type,
                'source_location': record.source_location,
                'created_at': record.created_at.isoformat(),
                'created_by': record.created_by,
                'record_count': record.record_count,
                'column_count': record.column_count,
                'schema': record.schema_info,
                'quality_metrics': record.quality_metrics,
                'transformation': record.transformation_applied,
                'parent_dataset_id': record.parent_dataset_id
            }
    
    def list_datasets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all registered datasets."""
        with get_db_session() as session:
            records = session.query(DataLineage).order_by(
                DataLineage.created_at.desc()
            ).limit(limit).all()
            
            return [
                {
                    'dataset_id': r.dataset_id,
                    'dataset_name': r.dataset_name,
                    'source_type': r.source_type,
                    'created_at': r.created_at.isoformat(),
                    'records': r.record_count
                }
                for r in records
            ]
    
    def generate_lineage_graph(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generate a graph structure showing dataset lineage.
        Useful for visualization.
        """
        nodes = []
        edges = []
        visited = set()
        
        def traverse(ds_id: str):
            if ds_id in visited or not ds_id:
                return
            
            visited.add(ds_id)
            
            with get_db_session() as session:
                record = session.query(DataLineage).filter_by(
                    dataset_id=ds_id
                ).first()
                
                if record:
                    nodes.append({
                        'id': record.dataset_id,
                        'name': record.dataset_name,
                        'type': record.source_type,
                        'created_at': record.created_at.isoformat()
                    })
                    
                    if record.parent_dataset_id:
                        edges.append({
                            'from': record.parent_dataset_id,
                            'to': record.dataset_id,
                            'transformation': record.transformation_applied
                        })
                        traverse(record.parent_dataset_id)
        
        traverse(dataset_id)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'root_dataset': dataset_id
        }


# Global instance
lineage_tracker = LineageTracker()
