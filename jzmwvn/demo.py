# this version is further changing demo.py, with weighted thresholding and taking into account fees.


# https://docs.qq.com/doc/DSUh0R0ZZeGdaSVRO
# 第二轮比赛demo - 使用CCXT获取Binance期货数据
import time
import os
import copy
import random
import pandas as pd
import numpy as np
from ccxt.pro import binance
from ccxt import binance as binance_sync
import logging
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict
from sdk.oms_client import OmsClient

# NEW !!! (1)
import xgboost as xgb
from joblib import load
import json
# (1) DONE



# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('strategy_btc_grid.log')
    ]
)
logger = logging.getLogger(__name__)

# 比赛环境会自动为策略加入代理IP， 以访止binance访问限频
if os.getenv('PROXY_IP'):
    proxies = {
        "http": os.getenv('PROXY_IP'),
        "https": os.getenv('PROXY_IP'),
    }
else:
    proxies = {}


class CryptoQuantDemo:
    def __init__(self):
        # Initilialise OMS client
        try:
            self.oms_client = OmsClient()
            logger.info("OMS客户端初始化成功")
        except Exception as e:
            logger.error(f"OMS客户端初始化失败: {e}")
            self.oms_client = None
        
        # Initialise CCXT exchange connections for Binance Futures
        try:
            self.exchange = binance({
                'proxies': proxies,
                'sandbox': False,  # 使用实盘数据
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 使用期货市场
                }
            })
            logger.info("CCXT Binance期货交易所初始化成功")

            self.exchange_sync = binance_sync({
                'proxies': proxies,
                'sandbox': False,  # 使用实盘数据
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # 使用期货市场
                }
            })
            logger.info("CCXT Binance期货交易所初始化成功")
        except Exception as e:
            logger.error(f"CCXT交易所初始化失败: {e}")
            self.exchange = None
        
        # Initialise data
        self.symbols = self.available_symbols
        self.historical_data = self.get_historical_data()
        self.account_balance = self.get_account_balance()
        self.current_positions = self.get_current_positions()
        self.target_positions = defaultdict(float)
        # self.position_switch_count = 2  # trade the top 2 and bottom 2 assets              NOT NEEDED DUE TO ADDITION OF THRESHOLDS LATER IN __init__
        # assert len(self.available_symbols) // 2 >= self.position_switch_count, "可交易品种数量不足"
        
        # Simulated time for testing
        self.current_time = datetime(2023, 10, 1, 7, 59)



        # NEW!!! (2), load pre-trained model and scaler
        try:
            self.model = xgb.Booster()
            self.model.load_model('trained_xgb_model.json')
            self.scaler = load('fitted_scaler.pkl')
            self.long_threshold = 0.02                             # MAY NEED TO CHANGE
            self.short_threshold = -0.02                           # MAY NEED TO CHANGE
            logger.info("Successfully loaded XGBoost model and scaler")
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load model artifacts: {e}")
            self.model = None
            self.scaler = None
        # (2) DONE

        # NEW!!! Define the same parameters used in training
        self.windows_1d = 4 * 24 * 1
        self.windows_7d = 4 * 24 * 7
        self.fast_macd, self.slow_macd, self.signal_macd = 12, 26, 9
        self.rsi_period, self.bb_period, self.ema_period = 14, 20, 20

        
    # 1. Available trading symbols
    @property
    def available_symbols(self):
        """可交易品种"""
        return [
            "BTC-USDT-PERP","ETH-USDT-PERP","SOL-USDT-PERP","BNB-USDT-PERP","XRP-USDT-PERP",
       #     "SOL-USDT-PERP", "DOT-USDT-PERP", "DOGE-USDT-PERP", "AVAX-USDT-PERP","XMR-USDT-PERP",
       #     "LTC-USDT-PERP", "LINK-USDT-PERP", "ATOM-USDT-PERP", "UNI-USDT-PERP", "XLM-USDT-PERP",
       #     "ALGO-USDT-PERP", "TRX-USDT-PERP", "ETC-USDT-PERP", "BCH-USDT-PERP",
        ]
    
    # 2. Get historical data, fetching last 7 days of 1-hour candles from Binance (used for 7-day return for momentum signal)
    def get_historical_data(self):
        """从Binance期货获取历史K线数据（7天）"""
        data = {}
        
        # 计算时间范围
        end_time = datetime.now()
        start_time = end_time - timedelta(days=7)
        since = int(start_time.timestamp() * 1000)  # 转换为毫秒时间戳
        
        for symbol in self.symbols:
            try:
                # 将 BTC-USDT-PERP 格式转换为 CCXT 格式 BTC/USDT:USDT
                base = symbol.split('-')[0]
                ccxt_symbol = f"{base}/USDT"
                
                # 获取1小时K线数据
                ohlcv = self.exchange_sync.fetch_ohlcv(ccxt_symbol, '15m', since=since, limit=672)  # 7天*24小时
                
                if not ohlcv:
                    logger.warning(f"未获取到 {symbol} 的历史数据，使用模拟数据")
                    data[symbol] = []
                    continue
                
                # 转换为DataFrame，保持原有格式
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                df["amount"] = df["volume"] * df["close"]
                
                data[symbol] = df
                logger.info(f"获取 {symbol} 历史数据: {len(df)} 条记录")
                
                # 避免请求过于频繁
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"获取 {symbol} 历史数据失败: {e}")
                data[symbol] = []
        
        logger.info(f"成功获取 {len(data)} 个品种的历史数据")
        return data
            
    
    # 3. 1min Kline callback (main bit), called every time a new 1min kline is received, checking if it's 08:00, if so, triggering run_strategy() to caclulate and place new trades
    async def on_1min_kline(self, symbol, kline):
        """
        1分钟K线回调函数
        :param symbol: 交易品种
        :param kline: K线数据 (dict格式)
        """
        # 这里可以添加策略逻辑
        kline_time = datetime.fromtimestamp(kline[0]/1000)
        logger.info(f"收到 {symbol} 1分钟K线[{kline_time.strftime('%Y-%m-%d %H:%M:%S')}]: "
              f"开={kline[1]:.4f}, 高={kline[2]:.4f}, "
              f"低={kline[3]:.4f}, 收={kline[4]:.4f}")
        
        # 检查是否到达策略执行时间（每天08:00）
        if self.current_time.hour == 8 and self.current_time.minute == 0:
            self.run_strategy()
            
    
    # 4. Query OMS client for current USDT balance (vital for position sizing)
    def get_account_balance(self):
        """从OMS客户端获取账户余额"""
        try:
            balances = self.oms_client.get_balance()
            balance_dict = {}
            for balance in balances:
                balance_dict[balance['asset']] = float(balance['balance'])
            
            print("账户余额:", balance_dict)
            return balance_dict
        except Exception as e:
            logger.error(f"获取账户余额失败: {e}")
            return {}
    
    # 5. Query OMS client for any open positions (vital for position sizing)
    def get_current_positions(self):
        """从OMS客户端获取当前持仓"""
        try:
            positions = self.oms_client.get_position()
            print("当前持仓:", positions)
            return positions
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            # 返回模拟数据作为备用
            return {}
    
    # 6. This is how strategy executes trades, taking dictionary of target positions, sends to OMS
    # For each symbol, calculates whether it's long or short and size in USDT, then sends order
    def push_target_positions(self, positions):
        """
        推送目标仓位
        :param positions: 目标仓位字典 {symbol: 目标价值(USDT)}
        """
        print("\n===== 推送目标仓位 =====")
        
        # 批量接口 限频1分钟一次
        # result = self.oms_client.set_target_position_batch([
        #   dict(
        #     instrument_name=instrument_name,
        #     instrument_type="future",
        #     target_value="%.2f" % abs_target_value,
        #     position_side=position_side
        #   )
        # ])
        
        for symbol, target_value in positions.items():
            try:
                instrument_name = symbol
                
                # 确定持仓方向和目标价值
                position_side = "LONG" if target_value >= 0 else "SHORT"
                abs_target_value = abs(target_value)
                
                # 如果目标价值为0，跳过
                if abs_target_value == 0:
                    print(f"{symbol}: 跳过 (目标价值为0)")
                    continue
                
                print(f"设置 {symbol}: {target_value:.2f} USDT ({position_side})")
                
                # 调用OMS客户端设置目标仓位
                result = self.oms_client.set_target_position(
                    instrument_name=instrument_name,
                    instrument_type="future",
                    target_value="%.2f" % abs_target_value,
                    position_side=position_side
                )
                
                logger.info(f"成功设置 {symbol} 目标仓位: {result}")

                self.account_balance = self.get_account_balance()
                self.current_positions = self.get_current_positions()
                
            except Exception as e:
                logger.error(f"设置 {symbol} 目标仓位失败: {e}")
                # 即使失败也保存到本地记录
                self.target_positions[symbol] = target_value
    
    # 7. 获取行情K线
    def get_ohlcv(self, symbol: str, timeframe: str, size: int):
        """
        获取Binance永续合约K线数据
        :param symbol: 交易对名称，永续合约格式，如 "BTCUSDT"
        :param timeframe: 时间周期，如 "1m", "5m", "1h", "1d" 等
            1m	1 分钟
            3m	3 分钟
            5m	5 分钟
            15m	15 分钟
            30m	30 分钟
            1h	1 小时
            2h	2 小时
            4h	4 小时
            6h	6 小时
            8h	8 小时
            12h	12 小时
            1d	1 天
            3d	3 天
            1w	1 周
            1M	1 个月
        :param size: 要获取的K线数量，最大不超过10000
        """
        try:
            ohlcvs = self.oms_client.fetch_ohlcv(symbol, timeframe, size)
            return ohlcvs
        except Exception as e:
            logger.error(f"获取K线失败: {e}")
            return []
    
    def show_account_detail(self):
        logger.info("==== 账户详情 ====")
        logger.info(f"账户余额: {self.account_balance}")
        logger.info(f"当前持仓: {self.current_positions}")

    # NEW !!!
    def _create_base_dataframes(self):
        close_data = []
        vwap_data = []
        amount_data = []
        index_ref = None
        for symbol in self.symbols:
            df = self.historical_data.get(symbol)
            if df is not None and not df.empty:
                df = df.iloc[-672:]  # Last 7 days of 15min data
                df_vwap_calc = (df['amount'] / df['volume']).replace([np.inf, -np.inf], np.nan).ffill()
                if index_ref is None:
                    index_ref = df.index
                    aligned_index = index_ref
                else:
                    aligned_index = index_ref.intersection(df.index)
                close_series = df['close'].reindex(aligned_index)
                vwap_series = df_vwap_calc.reindex(aligned_index)
                amount_series = df['amount'].reindex(aligned_index)

                close_data.append(close_series.rename(symbol))
                vwap_data.append(vwap_series.rename(symbol))
                amount_data.append(amount_series.rename(symbol))

        if not close_data:
            raise ValueError("Could not create base dataframes - no historical data available.")

        df_close = pd.concat(close_data, axis=1)
        df_vwap = pd.concat(vwap_data, axis=1)
        df_amount = pd.concat(amount_data, axis=1)

        df_close = df_close.ffill().dropna()
        df_vwap = df_vwap.ffill().dropna()
        df_amount = df_amount.ffill().dropna()

        logger.info(f"Created base dataframes with shape: Close {df_close.shape}, VWAP {df_vwap.shape}, Amount {df_amount.shape}")
        return df_close, df_vwap, df_amount




    # PASTE FUNCTIONS HEEEEERRRRRRREEEEE!!!!!!!!!!!!!!!!! (3)
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


    # "7d return calculation" no longer needed (part of old strategy) (4)
    
    # NEW!!! (4)
    def calculate_xgb_predictions(self):        
        """Calculate XGBoost predictions using 15m data"""
        if self.model is None or self.scaler is None:
            logger.error("Model or scaler not loaded. Cannot generate signals.")
            return {symbol: 0 for symbol in self.symbols}  # Fail safely in case model or scaler not loaded

        # ADDED CHANGES
        try:
            df_close, df_vwap, df_amount = self._create_base_dataframes()

            factors = {}
            current_time_index = df_vwap.index[-1]  
            all_symbol_list = self.symbols

            df_15min_rtn = df_vwap / df_vwap.shift(1) - 1
            
            print("Creating technical indicators...")
            for symbol in all_symbol_list:
                if symbol not in df_close.columns:
                    continue  # Skip if symbol wasn't in the aligned data

                factors.setdefault('rsi', pd.DataFrame(index=df_vwap.index, columns=all_symbol_list))
                factors['rsi'][symbol] = self.calculate_rsi(df_close[symbol]).shift(1)  # 1

                macd_line, signal_line, macd_hist = self.calculate_macd(df_close[symbol])
                factors.setdefault('macd_line', pd.DataFrame(index=df_vwap.index, columns=all_symbol_list))
                factors.setdefault('macd_signal', pd.DataFrame(index=df_vwap.index, columns=all_symbol_list))
                factors.setdefault('macd_histogram', pd.DataFrame(index=df_vwap.index, columns=all_symbol_list))
                factors['macd_line'][symbol] = macd_line.shift(1)  # 2
                factors['macd_signal'][symbol] = signal_line.shift(1)  # 3
                factors['macd_histogram'][symbol] = macd_hist.shift(1)  # 4

                factors.setdefault('ema', pd.DataFrame(index=df_vwap.index, columns=all_symbol_list))
                factors['ema'][symbol] = self.calculate_ema(df_close[symbol]).shift(1)  # 5

                _, _, bb_pos = self.calculate_bollinger_bands(df_close[symbol])
                factors.setdefault('bb_position', pd.DataFrame(index=df_vwap.index, columns=all_symbol_list))
                factors['bb_position'][symbol] = bb_pos.shift(1)  # 6

            print("Creating lagged returns...")
            windows_1d = 4 * 24 * 1  # 96 (15-min intervals in 1 day)
            windows_7d = 4 * 24 * 7  # 672 (15-min intervals in 7 days)
            factors['return_1h_lag1'] = df_vwap.shift(4) / df_vwap.shift(8) - 1  # 7
            factors['return_4h_lag1'] = df_vwap.shift(16) / df_vwap.shift(32) - 1  # 8
            factors['return_1d_lag1'] = df_vwap.shift(windows_1d) / df_vwap.shift(windows_1d * 2) - 1  # 9
            factors['return_7d_lag1'] = df_vwap.shift(windows_7d) / df_vwap.shift(windows_7d * 2) - 1  # 10

            print("Creating volatility features...")
            factors['volatility_7d_lag1'] = df_15min_rtn.rolling(windows_7d).std().shift(1)  # 11
            factors['volatility_24h'] = df_15min_rtn.rolling(96).std()  # 12

            print("Creating volume features...")
            factors['volume_avg_7d_lag1'] = df_amount.rolling(windows_7d).mean().shift(1)  # 13

            print("Creating momentum features...")
            factors['momentum_7d_lag1'] = (df_vwap / df_vwap.shift(windows_7d) - 1).shift(1)  # 14
            
            
            
            predictions = {}
            feature_names = [
                'rsi', 'macd_line', 'macd_signal', 'macd_histogram', 'ema', 'bb_position',
                'return_1h_lag1', 'return_4h_lag1', 'return_1d_lag1', 'return_7d_lag1',
                'volatility_7d_lag1', 'volatility_24h', 'volume_avg_7d_lag1', 'momentum_7d_lag1'
                ]

            for symbol in self.symbols:
                try:
                    feature_vector = []
                    for factor_name in feature_names:
                        # Get the value for this specific symbol and time
                        factor_df = factors.get(factor_name)
                        if factor_df is None or symbol not in factor_df.columns:
                            value = 0.0  # Default value if feature couldn't be calculated
                        else:
                            value = factor_df.loc[current_time_index, symbol]
                            if pd.isna(value):
                                value = 0.0
                        feature_vector.append(value)

                    
                    
                    # Scale features and predict
                    scaled_vector = self.scaler.transform([feature_vector])
                    dmatrix = xgb.DMatrix(scaled_vector)
                    predictions[symbol] = self.model.predict(dmatrix)[0]
                    logger.debug(f"Prediction for {symbol}: {predictions[symbol]:.6f}")

                except Exception as e:
                    logger.error(f"Error predicting for {symbol}: {e}")
                    predictions[symbol] = 0                               # sets asset to "no signal" and will not be traded as it will not be beyond thresholds
            
            return predictions
    
        except Exception as e:
            logger.error(f"Critical error in calculate_xgb_predictions: {e}")
            # Fail safely by returning no signal
            return {symbol: 0 for symbol in self.symbols}
    
    # (4) DONE




    # CORE STRATEGY: literally all trading logic here (called at 8:00 daily)
    # Calcuating which assets went up most and least in 7 days
    # Allocates capital: goes long on best performers and shorts worst performers (divides 90% of available cash equally between trades)
    # Executes trades via push_target_positions()

    # ADAPTED!!! (5)
    def run_strategy(self):
        """执行策略：多最强10个，空最弱10个"""
        print("\n===== Executing XGBoost Strategy =====")
        
        # NEW!!! Get predictions (instead of returns in old strategy)
        predictions = self.calculate_xgb_predictions()
        
        
        # No need to sort as we are using thresholds
        # sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        
        # The old selection of long and short symbols
        """long_symbols = [s for s, _ in sorted_returns[:self.position_switch_count]]
        short_symbols = [s for s, _ in sorted_returns[-self.position_switch_count:]]"""

        
        # NEW!!! Select symbols based on thresholds
        # ADAPTED: change to dictionary
        target_positions = {}
        long_symbols = {s: p for s, p in predictions.items() if p > self.long_threshold}
        short_symbols = {s: p for s, p in predictions.items() if p < self.short_threshold}

        print("Long Symbols:", long_symbols)
        print("Short Symbols:", short_symbols)
        
        # Calculate equal-weight position size (also added check to avoid division by zero))
        total_usdt = self.account_balance['USDT'] * 0.9

        # long and short allocation ratios based on strength of signals
        total_long_strength = sum(pred - self.long_threshold for pred in long_symbols.values())
        total_short_strength = sum(abs(pred) - abs(self.short_threshold) for pred in short_symbols.values())
        total_combined_strength = total_long_strength + total_short_strength
        
        if total_combined_strength > 0:
            long_allocation_ratio = total_long_strength / total_combined_strength
            short_allocation_ratio = total_short_strength / total_combined_strength
        else:
            long_allocation_ratio = 0.5 if long_symbols else 0
            short_allocation_ratio = 0.5 if short_symbols else 0
        

        # allocate to long positions
        if long_symbols:
            for symbol, pred in long_symbols.items():
                weight = (pred - self.long_threshold) / total_long_strength
                target_positions[symbol] = total_usdt * long_allocation_ratio * weight
        
        # Allocate to short positions
        if short_symbols:
            for symbol, pred in short_symbols.items():
                weight = (abs(pred) - abs(self.short_threshold)) / total_short_strength
                target_positions[symbol] = -total_usdt * short_allocation_ratio * weight
        
        # Apply transaction fee reduction
        fee_factor = 0.99954  # 1 - 0.046% round-trip fee
        target_positions = {s: p * fee_factor for s, p in target_positions.items()}

        # Push orders
        self.push_target_positions(target_positions)
    # (5) DONE



    # Process data
    async def _run(self):
        """运行策略"""
            
        # 订阅K线数据
        last_candle_time = 0
        last_candle = None
        watch_symbol = "BTC/USDT"
        self.last_show_info_time = datetime.now()
        while True:
            try:
                # 这里会阻塞直到收到K线数据, 每秒会更新多次kline
                ohlcv = await self.exchange.watch_ohlcv(watch_symbol, '15m')
            except Exception as e:
                logger.error(f"订阅K线数据错误: {e}")
                await asyncio.sleep(5)  # 等待5秒后重试
                continue

            try:
                if ohlcv and len(ohlcv) > 0:
                    curr_candle = ohlcv[-1]
                    if not last_candle:
                        last_candle = copy.copy(curr_candle)
                        last_candle_time = last_candle[0]
                        continue
                    if last_candle_time != curr_candle[0]:
                        last_candle_time = curr_candle[0]
                        await self.on_1min_kline(watch_symbol, last_candle)
                        # 每分钟更新一次当前时间
                        self.current_time = datetime.now()
                    last_candle = copy.copy(curr_candle)

                    # 每小时更新一次账户信息
                    if self.current_time - self.last_show_info_time > timedelta(hours=1):
                        self.account_balance = self.get_account_balance()
                        self.current_positions = self.get_current_positions()
                        self.last_show_info_time = self.current_time
                        self.show_account_detail()
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.exception(f"处理K线数据错误: {e}")
                await asyncio.sleep(5)  # 等待5秒后重试

    # Infinite loop to keep strategy running, handling exceptions and cleanup                
    async def run(self):
        try:
            await self._run()
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在关闭策略...")
        except Exception as e:
            logger.error(f"策略运行错误: {e}")
        finally:
            await self.cleanup()
    
    # Closing connections gracefully
    async def cleanup(self):
        """清理资源"""
        try:
            if self.exchange:
                await self.exchange.close()
            if self.oms_client:
                self.oms_client.close()
            logger.info("策略已停止")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
    
            

# Code that actually runs when executing script (initialising stategy, running once, creates endless)
if __name__ == "__main__":
    demo = CryptoQuantDemo()
    
    # 初始化后立即运行一次策略（可选）
    demo.run_strategy()
    demo.show_account_detail()

    asyncio.run(demo.run())
    