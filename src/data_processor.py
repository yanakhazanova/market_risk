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
    
    def rename_columns(self, data_frames):
        renamed_data_frames = {}
        for key, df in data_frames.items():
            new_columns = {col: f"{key}_{col}" for col in df.columns if col != 'date'}
            renamed_df = df.rename(columns=new_columns)
            renamed_data_frames[key] = renamed_df
        return renamed_data_frames
    
    def clean_data(self, df, row_threshold=0.25, col_threshold=0.25):
        # Удаление строк, если количество NaN превышает row_threshold
        row_thresh = int(row_threshold * df.shape[1])
        df_cleaned = df.dropna(thresh=row_thresh)
        
        # Удаление столбцов, если количество NaN превышает col_threshold
        col_thresh = int(col_threshold * df.shape[0])
        df_cleaned = df_cleaned.dropna(axis=1, thresh=col_thresh)
        
        return df_cleaned

    def fill_missing_data(self, df, method='ffill', value=None):
        if method == 'ffill':
            df_filled = df.ffill()
        elif method == 'bfill':
            df_filled = df.bfill()
        elif value is not None:
            df_filled = df.fillna(value)
        else:
            raise ValueError("Method must be 'ffill', 'bfill', or a specific value must be provided")
        return df_filled