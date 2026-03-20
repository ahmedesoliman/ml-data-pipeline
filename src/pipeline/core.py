"""
ML Data Pipeline Framework
Configurable data preprocessing and transformation
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional
import yaml


class DataTransformer(ABC):
    """Base class for data transformer operations"""
    
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe"""
        pass


class NormalizeTransformer(DataTransformer):
    """Normalize numeric columns to 0-1 range"""
    
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        cols_to_normalize = self.columns or df.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_normalize:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df


class StandardizeTransformer(DataTransformer):
    """Standardize numeric columns (zero mean, unit variance)"""
    
    def __init__(self, columns: Optional[List[str]] = None):
        self.columns = columns
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        cols_to_standardize = self.columns or df.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_standardize:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std != 0:
                    df[col] = (df[col] - mean) / std
        
        return df


class ImputeTransformer(DataTransformer):
    """Impute missing values"""
    
    def __init__(self, strategy: str = 'mean', value: Optional[Any] = None):
        """
        Args:
            strategy: 'mean', 'median', 'forward_fill', 'backward_fill', or 'value'
            value: Value to use with 'value' strategy
        """
        self.strategy = strategy
        self.value = value
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        if self.strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif self.strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif self.strategy == 'forward_fill':
            df = df.fillna(method='ffill')
        elif self.strategy == 'backward_fill':
            df = df.fillna(method='bfill')
        elif self.strategy == 'value' and self.value is not None:
            df = df.fillna(self.value)
        
        return df


class FilterTransformer(DataTransformer):
    """Filter rows based on conditions"""
    
    def __init__(self, conditions: Dict[str, tuple]):
        """
        Args:
            conditions: {'column_name': (operator, value)}
            operators: 'gt' (>), 'lt' (<), 'eq' (==), 'gte' (>=), 'lte' (<=)
        """
        self.conditions = conditions
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        for col, (operator, value) in self.conditions.items():
            if col not in df.columns:
                continue
            
            if operator == 'gt':
                df = df[df[col] > value]
            elif operator == 'lt':
                df = df[df[col] < value]
            elif operator == 'eq':
                df = df[df[col] == value]
            elif operator == 'gte':
                df = df[df[col] >= value]
            elif operator == 'lte':
                df = df[df[col] <= value]
        
        return df


class DropColumnsTransformer(DataTransformer):
    """Drop specified columns"""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop(columns=[col for col in self.columns if col in data.columns])


class SelectColumnsTransformer(DataTransformer):
    """Select only specified columns"""
    
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[[col for col in self.columns if col in data.columns]]


class DuplicateTransformer(DataTransformer):
    """Remove duplicate rows"""
    
    def __init__(self, keep: str = 'first'):
        """keep: 'first', 'last', or False (remove all duplicates)"""
        self.keep = keep
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.drop_duplicates(keep=self.keep)


class Pipeline:
    """Data preprocessing pipeline"""
    
    def __init__(self, transformers: List[DataTransformer] = None):
        self.transformers = transformers or []
    
    def add_transformer(self, transformer: DataTransformer) -> 'Pipeline':
        """Add a transformer to the pipeline"""
        self.transformers.append(transformer)
        return self
    
    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformers in sequence"""
        df = data.copy()
        for transformer in self.transformers:
            df = transformer.transform(df)
        return df
    
    def fit_from_config(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply transformations from configuration dictionary
        
        Example config:
        {
            'normalize': {'columns': ['age', 'income']},
            'impute': {'strategy': 'mean'},
            'filter': {'age': ('gte', 18)},
            'drop_columns': ['id', 'unused']
        }
        """
        data = config.pop('data', None)
        if data is None:
            raise ValueError("'data' key not found in config")
        
        df = data.copy()
        
        for transform_name, params in config.items():
            if transform_name == 'normalize':
                df = NormalizeTransformer(**params).transform(df)
            elif transform_name == 'standardize':
                df = StandardizeTransformer(**params).transform(df)
            elif transform_name == 'impute':
                df = ImputeTransformer(**params).transform(df)
            elif transform_name == 'filter':
                df = FilterTransformer(**params).transform(df)
            elif transform_name == 'drop_columns':
                df = DropColumnsTransformer(**params).transform(df)
            elif transform_name == 'select_columns':
                df = SelectColumnsTransformer(**params).transform(df)
            elif transform_name == 'duplicates':
                df = DuplicateTransformer(**params).transform(df)
        
        return df


class PipelineValidator:
    """Validate data quality"""
    
    @staticmethod
    def check_nulls(data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, float]:
        """Check null value percentages"""
        null_percentages = (data.isnull().sum() / len(data)) * 100
        return null_percentages[null_percentages > threshold].to_dict()
    
    @staticmethod
    def check_duplicates(data: pd.DataFrame) -> int:
        """Count duplicate rows"""
        return data.duplicated().sum()
    
    @staticmethod
    def check_data_types(data: pd.DataFrame) -> Dict[str, str]:
        """Get data types of columns"""
        return data.dtypes.astype(str).to_dict()
    
    @staticmethod
    def get_summary(data: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        return {
            'shape': data.shape,
            'null_values': data.isnull().sum().to_dict(),
            'duplicates': data.duplicated().sum(),
            'dtypes': data.dtypes.astype(str).to_dict(),
            'numeric_summary': data.describe().to_dict()
        }


def load_pipeline_config(yaml_file: str) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file"""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)
