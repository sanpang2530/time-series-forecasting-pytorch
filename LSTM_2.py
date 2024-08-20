import ccxt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 初始化Binance客户端
binance = ccxt.binance()


# 获取BTC/USDT的历史数据
def fetch_data(symbol='BTC/USDT', timeframe='1h', limit=500):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


data = fetch_data()

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
data['scaled_close'] = scaler.fit_transform(data['close'].values.reshape(-1, 1))


# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 初始化模型
model = LSTMModel()


# 训练数据准备
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append((seq, label))
    return np.array(sequences, dtype=object)  # 改进的地方


# 生成训练数据
seq_length = 10
sequences = create_sequences(data['scaled_close'].values, seq_length)

# 从 NumPy 数组转换为 Tensor
X = torch.tensor(np.stack(sequences[:, 0]), dtype=torch.float32).unsqueeze(-1)
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
    data_seq = torch.tensor(data[-seq_length:].reshape(1, -1, 1), dtype=torch.float32)

    with torch.no_grad():  # 禁用梯度计算
        prediction = model(data_seq)

    return prediction.item()


# 模拟持仓状态
current_position = None  # None, "long", "short"


# 模拟交易函数
def execute_trade(prediction, current_price, amount=0.001, symbol='BTC/USDT'):
    global current_position

    if prediction > current_price:
        if current_position == 'short':
            # 清空做空仓位
            print("清空做空仓位")
            binance.create_market_buy_order(symbol, amount)
            current_position = None
        if current_position is None:
            # 买入做多
            print(f"买入 {amount} BTC")
            binance.create_market_buy_order(symbol, amount)
            current_position = 'long'
    elif prediction < current_price:
        if current_position == 'long':
            # 清空做多仓位
            print("清空做多仓位")
            binance.create_market_sell_order(symbol, amount)
            current_position = None
        if current_position is None:
            # 卖出做空
            print(f"卖出 {amount} BTC")
            binance.create_market_sell_order(symbol, amount)
            current_position = 'short'


# 实时预测与交易
def trade():
    current_price = data['close'].iloc[-1]
    prediction = predict(model, data['scaled_close'].values, seq_length)
    predicted_price = scaler.inverse_transform([[prediction]])[0, 0]
    print(f"当前价格: {current_price}, 预测价格: {predicted_price}")

    execute_trade(predicted_price, current_price)


trade()