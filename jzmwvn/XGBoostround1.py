import numpy as np
import pandas as pd
import datetime, os, time
import multiprocessing as mp
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import gc  # Garbage collection

import joblib  # NEW!!!

class XGBModel:
    def __init__(self):
        self.train_data_path = "/Users/patrick/QRT-HKU/hku-comp-qrt/avenir_hku_web_data/kline_data/train_data"
        self.start_datetime = datetime.datetime(2021, 3, 1, 0, 0, 0)
        self.scaler = StandardScaler()
    
    def get_all_symbol_list(self):
        parquet_name_list = os.listdir(self.train_data_path)
        symbol_list = [f.split(".")[0] for f in parquet_name_list if f.endswith(".parquet")]
        return symbol_list
    
    def get_single_symbol_kline_data(self, symbol):
        try:
            df = pd.read_parquet(f"{self.train_data_path}/{symbol}.parquet")
            df = df.set_index("timestamp")
            df = df.astype(np.float64)
            df['vwap'] = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).ffill()
        except Exception as e:
            print(f"get_single_symbol_kline_data error: {e}")
            df = pd.DataFrame()
        return df
    
    def get_all_symbol_kline(self):
        t0 = datetime.datetime.now()
        pool = mp.Pool(mp.cpu_count() - 2)
        all_symbol_list = self.get_all_symbol_list()
        df_list = []
        
        for symbol in all_symbol_list:
            df_list.append(pool.apply_async(self.get_single_symbol_kline_data, (symbol,)))
        
        pool.close()
        pool.join()
        
        # Get results
        results = [r.get() for r in df_list]
        df_open_price = pd.concat([r['open_price'] for r in results], axis=1).sort_index(ascending=True)
        time_arr = pd.to_datetime(pd.Series(df_open_price.index), unit="ms").values
        
        # Build arrays efficiently
        vwap_arr = pd.concat([r['vwap'] for r in results], axis=1).sort_index(ascending=True).values
        amount_arr = pd.concat([r['amount'] for r in results], axis=1).sort_index(ascending=True).values
        close_arr = pd.concat([r['close_price'] for r in results], axis=1).sort_index(ascending=True).values
        
        print(f"finished get all symbols kline, time escaped {datetime.datetime.now() - t0}")
        return all_symbol_list, time_arr, vwap_arr, amount_arr, close_arr
    
    def weighted_spearmanr(self, y_true, y_pred):
        n = len(y_true)
        r_true = pd.Series(y_true).rank(ascending=False, method='average')
        r_pred = pd.Series(y_pred).rank(ascending=False, method='average')
        
        x = 2 * (r_true - 1) / (n - 1) - 1
        w = x ** 2  
        
        w_sum = w.sum()
        mu_true = (w * r_true).sum() / w_sum
        mu_pred = (w * r_pred).sum() / w_sum
        
        cov = (w * (r_true - mu_true) * (r_pred - mu_pred)).sum()
        var_true = (w * (r_true - mu_true)**2).sum()
        var_pred = (w * (r_pred - mu_pred)**2).sum()
        
        return cov / np.sqrt(var_true * var_pred)

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, min_periods=1).mean()
        ema_slow = prices.ewm(span=slow, min_periods=1).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=1).mean()
        return macd_line, signal_line, macd_line - signal_line

    def calculate_ema(self, prices, period=20):
        return prices.ewm(span=period, min_periods=1).mean()

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        sma = prices.rolling(window=period, min_periods=1).mean()
        std = prices.rolling(window=period, min_periods=1).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        bb_position = (prices - lower_band) / (upper_band - lower_band).replace(0, 1e-10)
        return upper_band, lower_band, bb_position

    def train_memory_efficient(self, df_target, df_factors):
        print("Preparing data for memory-efficient training...")
        
        # Create target series
        target_long = df_target.stack()
        target_long.name = 'target'
        
        # Create features with proper alignment
        feature_data = []
        for factor_name, factor_df in df_factors.items():
            # Align with target index first
            aligned_factor = factor_df.reindex(df_target.index)
            factor_long = aligned_factor.stack()
            factor_long.name = factor_name
            feature_data.append(factor_long)
        
        # Merge and clean
        data = pd.concat([target_long] + feature_data, axis=1)
        data = data.dropna()
        
        if data.empty:
            raise ValueError("No data available after cleaning!")
        
        print(f"Final training data shape: {data.shape}")
        
        # Use only top 10 features to save memory
        feature_cols = [col for col in data.columns if col != 'target']
        X = data[feature_cols]
        y = data['target']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train final model with memory-efficient settings
        print("Training final model...")
        xgb_model = xgb.XGBRegressor(
            tree_method='hist',
            n_estimators=300,  # Reduced for memory
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_model.fit(X_scaled, y)
        
        # Create submission in memory-efficient way
        print("Creating submission...")
        data['y_pred'] = xgb_model.predict(X_scaled)
        
        # Prepare submission
        #submission = data.reset_index()[['level_0', 'level_1', 'y_pred']]
        #submission.columns = ['datetime', 'symbol', 'predict_return']
        #submission = submission[submission['datetime'] >= self.start_datetime]
        #submission['id'] = submission['datetime'].astype(str) + "_" + submission['symbol']
        
        # Load submission template
        #submission_template = pd.read_csv("/tmp/submission_id.csv")
        
        # Merge with template
        #final_submission = submission_template.merge(
        #    submission[['id', 'predict_return']], 
        #    on='id', 
        #    how='left'
        #).fillna(0)
        
        #final_submission.to_csv("final_submission_JZMWVN.csv", index=False)
        
        #print(f"Submission created with shape: {final_submission.shape}")
        #print(f"Non-zero predictions: {(final_submission['predict_return'] != 0).sum()}")
        
        # Calculate performance
        #spearman = self.weighted_spearmanr(data['target'], data['y_pred'])
        #print(f"Final model Spearman: {spearman:.4f}")


        # NEW!!!
        xgb_model.save_model('jzmwvn/trained_xgb_model.json')
        joblib.dump(self.scaler, 'jzmwvn/fitted_scaler.pkl')
        print("Model and scaler saved successfully!")
        
        return xgb_model

    def run(self):
        # Get data
        all_symbol_list, time_arr, vwap_arr, amount_arr, close_arr = self.get_all_symbol_kline()
        
        # Convert to DataFrames
        df_vwap = pd.DataFrame(vwap_arr, columns=all_symbol_list, index=time_arr)
        df_amount = pd.DataFrame(amount_arr, columns=all_symbol_list, index=time_arr)
        df_close = pd.DataFrame(close_arr, columns=all_symbol_list, index=time_arr)
        
        # Define time windows
        windows_1d = 4 * 24 * 1
        windows_7d = 4 * 24 * 7
        
        # Calculate target
        target = df_vwap.shift(-windows_7d) / df_vwap - 1
        
        # Calculate 15-min returns for volatility
        df_15min_rtn = df_vwap / df_vwap.shift(1) - 1
        
        # Create features (your excellent feature engineering)
        factors = {}
        
        print("Creating technical indicators...")
        for symbol in all_symbol_list:
            factors.setdefault('rsi', pd.DataFrame(index=time_arr, columns=all_symbol_list))
            factors['rsi'][symbol] = self.calculate_rsi(df_close[symbol]).shift(1)
            
            macd_line, signal_line, macd_hist = self.calculate_macd(df_close[symbol])
            factors.setdefault('macd_line', pd.DataFrame(index=time_arr, columns=all_symbol_list))
            factors.setdefault('macd_signal', pd.DataFrame(index=time_arr, columns=all_symbol_list))
            factors.setdefault('macd_histogram', pd.DataFrame(index=time_arr, columns=all_symbol_list))
            factors['macd_line'][symbol] = macd_line.shift(1)
            factors['macd_signal'][symbol] = signal_line.shift(1)
            factors['macd_histogram'][symbol] = macd_hist.shift(1)
            
            factors.setdefault('ema', pd.DataFrame(index=time_arr, columns=all_symbol_list))
            factors['ema'][symbol] = self.calculate_ema(df_close[symbol]).shift(1)
            
            _, _, bb_pos = self.calculate_bollinger_bands(df_close[symbol])
            factors.setdefault('bb_position', pd.DataFrame(index=time_arr, columns=all_symbol_list))
            factors['bb_position'][symbol] = bb_pos.shift(1)
        
        print("Creating lagged returns...")
        factors['return_1h_lag1'] = df_vwap.shift(4) / df_vwap.shift(8) - 1
        factors['return_4h_lag1'] = df_vwap.shift(16) / df_vwap.shift(32) - 1
        factors['return_1d_lag1'] = df_vwap.shift(windows_1d) / df_vwap.shift(windows_1d * 2) - 1
        factors['return_7d_lag1'] = df_vwap.shift(windows_7d) / df_vwap.shift(windows_7d * 2) - 1
        
        print("Creating volatility features...")
        factors['volatility_7d_lag1'] = df_15min_rtn.rolling(windows_7d).std().shift(1)
        factors['volatility_24h'] = df_15min_rtn.rolling(96).std()
        
        print("Creating volume features...")
        factors['volume_avg_7d_lag1'] = df_amount.rolling(windows_7d).mean().shift(1)
        
        print("Creating momentum features...")
        factors['momentum_7d_lag1'] = (df_vwap / df_vwap.shift(windows_7d) - 1).shift(1)
        
        # Clean up factors
        for factor_name in list(factors.keys()):
            factors[factor_name] = factors[factor_name].dropna(how='all')
        
        print("Training model...")
        self.train_memory_efficient(target, factors)
        
if __name__ == '__main__':
    script_start = datetime.datetime.now()
    model = XGBModel()
    model.run()
    script_end = datetime.datetime.now()
    print("Total script run time:", script_end - script_start)