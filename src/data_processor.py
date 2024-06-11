import os
import pandas as pd
import pandas_market_calendars as mcal

class DataProcessor:
    def __init__(self, ):
        pass

    def load_data(self, file_paths):
        data_frames = {}
        for key, path in file_paths.items():
            df = pd.read_csv(path, parse_dates=['date'])
            data_frames[key] = pd.merge(self.trading_days, df, on='date', how='left')
            data_frames[key].fillna(method='ffill', inplace=True)
            data_frames[key].interpolate(method='linear', inplace=True)
        return data_frames

    def merge_data(self, data_frames):
        all_data = self.trading_days
        for key, df in data_frames.items():
            all_data = pd.merge(all_data, df, on='date', how='outer')
        all_data.fillna(method='ffill', inplace=True)
        return all_data
