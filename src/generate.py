import pandas as pd
import numpy as np

# Assume we have 5 features and 1 label (fraud or not fraud)
num_samples = 10000
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, num_samples),
    'feature2': np.random.normal(0, 2, num_samples),
    'feature3': np.random.normal(0, 3, num_samples),
    'feature4': np.random.normal(0, 4, num_samples),
    'feature5': np.random.normal(0, 5, num_samples),
    'label': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])  # 5% fraud
})
data.to_csv('data/raw/dummy_data.csv', index=False)

