import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/raw/dummy_data.csv')
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(columns=['label']))
data_scaled = pd.DataFrame(data_scaled, columns=data.columns[:-1])
data_scaled['label'] = data['label']
data_scaled.to_csv('data/processed/dummy_data.csv', index=False)

