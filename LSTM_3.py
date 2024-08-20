import ccxt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import talib as ta  # 引入 TA-Lib 计算技术指标
import time

# 初始化Binance客户端
binance = ccxt.binance()


# 获取BTC/USDT的历史数据
def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=500):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


data = fetch_data()


# 计算技术指标
def calculate_technical_indicators(df):
    df['SMA'] = ta.SMA(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['RSI'] = ta.RSI(df['close'], timeperiod=14)
    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['Volume'] = df['volume']  # 引入成交量

    # 处理缺失值
    df.fillna(0, inplace=True)

    return df


data = calculate_technical_indicators(data)

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
data[['scaled_close', 'scaled_SMA', 'scaled_MACD', 'scaled_RSI', 'scaled_Upper_BB', 'scaled_Lower_BB',
      'scaled_Volume']] = scaler.fit_transform(
    data[['close', 'SMA', 'MACD', 'RSI', 'Upper_BB', 'Lower_BB', 'Volume']]
)


# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_size=7, d_model=64, nhead=8, num_layers=2, output_size=1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, -1, :])  # 只取最后一个时间步的输出
        return x


# 初始化Transformer模型
model = TransformerModel()


# 训练数据准备
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length, :-1]
        label = data[i + seq_length, -1]
        sequences.append((seq, label))
    return np.array(sequences, dtype=object)


# 特征包括收盘价、SMA、MACD、RSI、布林带上下轨和成交量
seq_length = 10
features = data[['scaled_close', 'scaled_SMA', 'scaled_MACD', 'scaled_RSI', 'scaled_Upper_BB', 'scaled_Lower_BB',
                 'scaled_Volume']].values
sequences = create_sequences(features, seq_length)

# 从 NumPy 数组转换为 Tensor
X = torch.tensor(np.stack(sequences[:, 0]), dtype=torch.float32)
y = torch.tensor(np.stack(sequences[:, 1]), dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 10
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


# 预测函数
def predict(model, data, seq_length):
    model.eval()  # 设置模型为评估模式
    data_seq = torch.tensor(data[-seq_length:, :-1].reshape(1, seq_length, -1), dtype=torch.float32)

    with torch.no_grad():  # 禁用梯度计算
        prediction = model(data_seq)

    return prediction.item()


# 模拟持仓状态
current_position = None  # None, "long", "short"
last_trade_time = None  # 记录最后一次交易时间


# 模拟交易函数，增加阈值和冷却期逻辑
def execute_trade(prediction, current_price, amount=0.001, symbol='BTC/USDT', threshold=0.005, cooldown=3600):
    global current_position
    global last_trade_time

    # 获取当前时间
    current_time = time.time()

    # 确保在冷却期内不交易
    if last_trade_time is not None and current_time - last_trade_time < cooldown:
        print("冷却期未过，不执行交易。")
        return

    # 计算预测价格与当前价格的相对变化
    price_change = abs(prediction - current_price) / current_price

    # 如果价格变化小于阈值，则不执行交易
    if price_change < threshold:
        print(f"价格变化 {price_change:.4f} 小于阈值 {threshold}，不执行交易。")
        return

    if prediction > current_price:
        if current_position == 'short':
            # 清空做空仓位
            print("清空做空仓位")
            # binance.create_market_buy_order(symbol, amount)
            current_position = None
        if current_position is None:
            # 买入做多
            print(f"买入 {amount} BTC")
            # binance.create_market_buy_order(symbol, amount)
            current_position = 'long'
            last_trade_time = current_time  # 记录最后一次交易时间
    elif prediction < current_price:
        if current_position == 'long':
            # 清空做多仓位
            print("清空做多仓位")
            binance.create_market_sell_order(symbol, amount)
            current_position = None
        if current_position is None:
            # 卖出做空
            print(f"卖出 {amount} BTC")
            # binance.create_market_sell_order(symbol, amount)
            current_position = 'short'
            last_trade_time = current_time  # 记录最后一次交易时间


# 实时预测与交易
def trade():
    current_price = data['close'].iloc[-1]
    prediction = predict(model, features, seq_length)
    predicted_price = scaler.inverse_transform([[prediction, 0, 0, 0, 0, 0, 0]])[0, 0]
    print(f"当前价格: {current_price}, 预测价格: {predicted_price}")

    # 调整阈值为 0.5% 的价格变化，冷却时间为 1 小时（3600 秒）
    execute_trade(predicted_price, current_price, amount=0.001, threshold=0.005, cooldown=3600)


if __name__ == '__main__':
    trade()
