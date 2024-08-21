import ccxt
import pandas as pd
import ta
import torch
""" 工具函数 (utils.py) """

# Binance API 配置
exchange = ccxt.binance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
})

def fetch_ohlcv(symbol, timeframe='1m', limit=100):
    """获取币安市场的K线数据"""
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def compute_indicators(df):
    """计算MACD和RSI指标"""
    df['MACD'] = ta.trend.MACD(df['close']).macd()
    df['MACD_signal'] = ta.trend.MACD(df['close']).macd_signal()
    df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
    return df

def save_model(model, path='data/saved_model.pth'):
    """保存模型"""
    torch.save(model.state_dict(), path)

def load_model(model_class, path='data/saved_model.pth'):
    """加载模型"""
    model = model_class()
    model.load_state_dict(torch.load(path))
    return model