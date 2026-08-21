"""
config.py

Centralized constants for file paths, target column, and model
training parameters, shared across the pipeline's modules.
"""

RAW_DATA_PATH = 'data/raw/housing.csv'
TARGET_COL = 'MedHouseVal'
CAP_THRESHOLD = 5.0
TEST_SIZE = 0.2
RANDOM_STATE = 42