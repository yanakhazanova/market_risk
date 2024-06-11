import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAAnalyzer:
    def __init__(self, data):
        self.data = data
        self.scaled_data = self.scale_data(data)
        self.pca = PCA()

    def scale_data(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))
        return scaled_data

    def perform_pca(self, n_components):
        self.pca = PCA(n_components=n_components)
        principal_components = self.pca.fit_transform(self.scaled_data)
        return pd.DataFrame(data=principal_components)

    def plot_correlation_matrix(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(pd.DataFrame(self.scaled_data).corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    def descriptive_statistics(self):
        stats = pd.DataFrame(data=self.scaled_data).describe()
        stats.loc['skew'] = pd.DataFrame(data=self.scaled_data).skew()
        stats.loc['kurtosis'] = pd.DataFrame(data=self.scaled_data).kurtosis()
        return stats
