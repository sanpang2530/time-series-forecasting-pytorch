import time
from datetime import datetime

import torch
from torch import nn, optim

from TransformerLibs import TransformerModel
from Transformer_2_predict import fetch_binance_data


# 增量训练函数
def incremental_train(model, new_data, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(new_data)
        loss = criterion(output, new_data[:, -1, :])
        loss.backward()
        optimizer.step()
        print(f'Incremental training loss: {loss.item()}')


# 实时训练和模型保存
def real_time_training_loop(model, criterion, optimizer, seq_length=100, save_interval=3600):
    last_save_time = time.time()

    while True:
        # 获取实时数据并训练
        new_data = fetch_binance_data()[-seq_length:].reshape(1, seq_length, -1)
        new_data = torch.tensor(new_data, dtype=torch.float32)

        incremental_train(model, new_data, criterion, optimizer)

        # 检查是否该保存模型
        current_time = time.time()
        if current_time - last_save_time >= save_interval:
            save_path = f'./transformer_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            torch.save(model.state_dict(), save_path)
            print(f'Model saved to {save_path}')
            last_save_time = current_time


# 初始化模型、损失函数和优化器
model = TransformerModel(input_size=7, d_model=64, nhead=8, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 实时训练循环
real_time_training_loop(model, criterion, optimizer, seq_length=100)