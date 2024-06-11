import pandas as pd

class DataProcessor:
    def __init__(self, trading_days, start_date=None, end_date=None):
        if trading_days is not None:
            self.trading_days = trading_days
            if start_date is not None and end_date is not None:
                self.start_date = start_date
                self.end_date = end_date
            else:
                self.start_date = min(trading_days['date'])
                self.end_date = max(trading_days['date'])
                
    def load_data(self, file_paths):
        data_frames = {}
        for key, path in file_paths.items():
            df = pd.read_csv(path, parse_dates=['date'])
            merged_df = pd.merge(self.trading_days, df, on='date', how='left')
            merged_df.ffill(inplace=True)
            merged_df.interpolate(method='linear', inplace=True)
            data_frames[key] = merged_df
        return data_frames

    def merge_data(self, data_frames):
        all_data = self.trading_days
        for key, df in data_frames.items():
            all_data = pd.merge(all_data, df, on='date', how='outer')
        all_data.ffill(inplace=True)
        return all_data
