import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/processed/dummy_data.csv')

sns.pairplot(data, hue='label')
plt.savefig('pairplot.png')

# Plot correlations
corr = data.corr()
sns.heatmap(corr, annot=True)
plt.savefig('correlation_matrix.png')

