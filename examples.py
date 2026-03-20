"""Example usage of ML Data Pipeline"""
import pandas as pd
from pipeline.core import Pipeline, PipelineValidator, NormalizeTransformer, ImputeTransformer, FilterTransformer
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
    
    # Visualize before and after
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    data.plot(kind='bar', ax=axes[0], title='Original Data')
    result.plot(kind='bar', ax=axes[1], title='After Imputation & Normalization')
    plt.tight_layout()
    plt.savefig('output.png')
    plt.close()


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


def example_learning_with_more_data():
    """Example 4: Learning with more data (regression)"""
    print("\n" + "=" * 60)
    print("Example 4: Learning with More Data (Regression)")
    print("=" * 60)
    np.random.seed(42)
    # Generate synthetic data
    n_samples = 1000
    age = np.random.randint(18, 70, n_samples).astype(float)
    score = np.random.normal(75, 10, n_samples)
    # True relationship: income = 1000 * age + 500 * score + noise
    income = 1000 * age + 500 * score + np.random.normal(0, 10000, n_samples)
    # Introduce some missing values
    mask = np.random.rand(n_samples) < 0.05
    age[mask] = np.nan
    mask = np.random.rand(n_samples) < 0.05
    score[mask] = np.nan
    mask = np.random.rand(n_samples) < 0.05
    income[mask] = np.nan
    data = pd.DataFrame({'age': age, 'score': score, 'income': income})
    print("\nSample of Original Data:")
    print(data.head())
    # Pipeline: impute and normalize
    pipeline = Pipeline()
    pipeline.add_transformer(ImputeTransformer(strategy='mean'))
    pipeline.add_transformer(NormalizeTransformer())
    processed = pipeline.fit(data)
    print("\nSample After Imputation & Normalization:")
    print(processed.head())
    # ML: Predict income from age and score
    X = processed[['age', 'score']]
    y = processed['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    score_val = model.score(X_test, y_test)
    print(f"\nLinear Regression R^2 on test set: {score_val:.3f}")
    print(f"Coefficients: age={model.coef_[0]:.3f}, score={model.coef_[1]:.3f}, intercept={model.intercept_:.3f}")
    # Visualization: true vs predicted
    y_pred = model.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('True Income (normalized)')
    plt.ylabel('Predicted Income (normalized)')
    plt.title('Regression: True vs Predicted Income')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.tight_layout()
    plt.savefig('regression_output.png')
    plt.close()


if __name__ == "__main__":
    import numpy as np
    example_basic_pipeline()
    example_filtering()
    example_validation()
    example_learning_with_more_data()
