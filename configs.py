"""Pipeline configuration examples as YAML
Usage: Load with load_pipeline_config('config.yaml')
"""

# Example 1: Basic preprocessing
basic_config = """
transformers:
  - type: impute
    strategy: mean
  - type: normalize
    columns: [age, salary]
  - type: drop_columns
    columns: [id, internal_notes]
"""

# Example 2: Customer data pipeline
customer_config = """
transformers:
  - type: filter
    conditions:
      age: [gte, 18]
      status: [eq, 'active']
  - type: impute
    strategy: median
  - type: duplicates
    keep: first
  - type: normalize
    columns: [lifetime_value, purchase_count]
"""

# Example 3: Financial data pipeline
financial_config = """
transformers:
  - type: drop_columns
    columns: [internal_id, notes]
  - type: impute
    strategy: forward_fill
  - type: filter
    conditions:
      transaction_amount: [gt, 0]
  - type: standardize
    columns: [price, volume, returns]
"""
