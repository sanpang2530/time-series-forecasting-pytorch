import torch
import requests
import numpy as np

from TransformerLibs import TransformerModel


# 从币安API获取实时数据
def fetch_binance_data():
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': 100  # 获取100条数据
    }
    response = requests.get(url, params=params)
    data = response.json()
    # 数据处理，提取所需特征（例如收盘价）
    close_prices = [float(kline[4]) for kline in data]  # 4是收盘价
    return np.array(close_prices)


# 预测函数
def predict_price(model, seq_length=100):
    model.eval()
    with torch.no_grad():
        # 获取实时数据
        real_time_data = fetch_binance_data()
        # 进行预处理，将其转换为 Tensor
        real_time_data = real_time_data[-seq_length:].reshape(1, seq_length, -1)
        real_time_data = torch.tensor(real_time_data, dtype=torch.float32)

        # 使用模型进行预测
        prediction = model(real_time_data)
        print(f'Predicted price: {prediction.item()}')


# 加载保存的模型
def load_model(model_path='./transformer_model.pth'):
    model = TransformerModel(input_size=7, d_model=64, nhead=8, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 加载模型并预测
model = load_model()
predict_price(model)