import pandas as pd
import numpy as np
import sys

# add your src directory to the path
sys.path.append("../src")
import data.preprocess as preprocess

def test_preprocessing():
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 10),
        'feature2': np.random.normal(0, 2, 10),
        'feature3': np.random.normal(0, 3, 10),
        'feature4': np.random.normal(0, 4, 10),
        'feature5': np.random.normal(0, 5, 10),
        'label': np.random.choice([0, 1], 10)
    })

    processed_df = preprocess(df)

    assert processed_df.isnull().sum().sum() == 0, "There should be no null values."
    assert processed_df.shape == df.shape, "Input and output shape should be the same."
    assert processed_df['label'].equals(df['label']), "Labels should not be changed."

