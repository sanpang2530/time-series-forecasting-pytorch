import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os

from TransformerLibs import TransformerModel, train_loader


# 假设之前定义的 TransformerModel 类已经存在

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')

    # 保存模型
    model_save_path = './transformer_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

# 初始化模型、损失函数和优化器
model = TransformerModel(input_size=7, d_model=64, nhead=8, num_layers=2, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 假设 train_loader 已经准备好
train_model(model, train_loader, criterion, optimizer, epochs=10)