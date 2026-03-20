"""Example usage of ML Data Pipeline"""
import pandas as pd
from pipeline.core import Pipeline, PipelineValidator, NormalizeTransformer, ImputeTransformer, FilterTransformer


def example_basic_pipeline():
    """Example 1: Basic pipeline with transformers"""
    print("=" * 60)
    print("Example 1: Basic Pipeline")
    print("=" * 60)
    
    # Create sample data
    data = pd.DataFrame({
        'age': [25, 30, np.nan, 45, 50],
        'income': [30000, 50000, 60000, 80000, np.nan],
        'score': [65, 70, 85, 90, 88]
    })
    
    print("\nOriginal Data:")
    print(data)
    
    # Create pipeline
    pipeline = Pipeline()
    pipeline.add_transformer(ImputeTransformer(strategy='mean'))
    pipeline.add_transformer(NormalizeTransformer())
    
    result = pipeline.fit(data)
    print("\nAfter Imputation & Normalization:")
    print(result)


def example_filtering():
    """Example 2: Pipeline with filtering"""
    print("\n" + "=" * 60)
    print("Example 2: Pipeline with Filtering")
    print("=" * 60)
    
    data = pd.DataFrame({
        'age': [15, 22, 45, 18, 65],
        'income': [0, 30000, 90000, 25000, 120000],
        'region': ['US', 'US', 'EU', 'US', 'EU']
    })
    
    print("\nOriginal Data:")
    print(data)
    
    # Filter for adults with income > 25000
    pipeline = Pipeline()
    pipeline.add_transformer(FilterTransformer(conditions={'age': ('gte', 18)}))
    pipeline.add_transformer(FilterTransformer(conditions={'income': ('gt', 25000)}))
    
    result = pipeline.fit(data)
    print("\nFiltered: age >= 18 AND income > 25000:")
    print(result)


def example_validation():
    """Example 3: Data validation"""
    print("\n" + "=" * 60)
    print("Example 3: Data Validation")
    print("=" * 60)
    
    data = pd.DataFrame({
        'id': [1, 2, 2, 4, 5],
        'name': ['Alice', 'Bob', 'Bob', 'David', None],
        'score': [85, 90, 90, 78, 88]
    })
    
    print("\nData Summary:")
    validator = PipelineValidator()
    summary = validator.get_summary(data)
    print(f"Shape: {summary['shape']}")
    print(f"Null values: {summary['null_values']}")
    print(f"Duplicates: {summary['duplicates']}")


if __name__ == "__main__":
    import numpy as np
    
    example_basic_pipeline()
    example_filtering()
    example_validation()
