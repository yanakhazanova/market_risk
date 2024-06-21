import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis, skew

class RiskFactorAnalyzer:
    def __init__(self, data):
        self.data = data
        self.pca = None
        self.explained_variance_ratio_ = None
        self.pca_components_ = None

    def perform_pca(self, n_components=None):
        # Scale the data
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.data.drop(columns=['date']))
        
        # Perform PCA
        self.pca = PCA(n_components=n_components)
        self.pca_components_ = self.pca.fit_transform(self.scaled_data)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
    
    def plot_pca_variance(self):
        plt.figure(figsize=(10, 4))
        plt.plot(np.cumsum(self.explained_variance_ratio_), marker='o', linestyle='--')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance by PCA Components')
        plt.grid(True)
        plt.show()

    def visualize_risk_factors(self):
        for column in self.data.columns:
            if column != 'date':
                plt.figure(figsize=(10, 4))
                plt.plot(self.data['date'], self.data[column], label=column)
                plt.xlabel('Date')
                plt.ylabel(column)
                plt.title(f'Time Series of {column}')
                plt.legend()
                plt.grid(True)
                plt.show()

    def visualize_histograms(self):
        for column in self.data.columns:
            if column != 'date':
                plt.figure(figsize=(10, 4))
                sns.histplot(self.data[column], kde=True)
                plt.title(f'Histogram of {column}')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.show()
                
    def visualize_boxplots(self):
        for column in self.data.columns:
            if column != 'date':
                plt.figure(figsize=(10, 4))
                sns.boxplot(x=self.data[column])
                plt.title(f'Boxplot of {column}')
                plt.ylabel('Value')
                plt.show()

    def plot_correlation_matrix(self):
        correlation_matrix = self.data.drop(columns=['date']).corr()
        plt.figure(figsize=(9, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Risk Factors')
        plt.show()

    def descriptive_statistics(self):
        desc_stats = self.data.drop(columns=['date']).describe().T
        desc_stats['kurtosis'] = self.data.drop(columns=['date']).kurtosis()
        desc_stats['skewness'] = self.data.drop(columns=['date']).skew()
        return desc_stats

    def trend_seasonality_stationarity(self, column):
        decomposition = seasonal_decompose(self.data.set_index('date')[column], model='additive', period=252)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(14, 10))
        plt.subplot(411)
        plt.plot(self.data['date'], self.data[column], label='Original')
        plt.legend(loc='upper left')
        plt.subplot(412)
        plt.plot(self.data['date'], trend, label='Trend')
        plt.legend(loc='upper left')
        plt.subplot(413)
        plt.plot(self.data['date'], seasonal, label='Seasonality')
        plt.legend(loc='upper left')
        plt.subplot(414)
        plt.plot(self.data['date'], residual, label='Residuals')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    def handle_missing_data(self):
        # Удаляем столбцы с более чем 50% пропущенных значений
        self.data = self.data.loc[:, self.data.isnull().mean() < 0.5]
        # Удаляем строки с более чем 50% пропущенных значений
        self.data.dropna(thresh=int(self.data.shape[1] * 0.5), inplace=True)
        # Заполняем оставшиеся пропуски линейной интерполяцией
        self.data.interpolate(method='linear', inplace=True)

    def fill_missing_data(self):
        # Заполнение оставшихся пропусков
        self.data.fillna(method='ffill', inplace=True)
        self.data.dropna()

