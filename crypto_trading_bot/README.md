# Crypto Trading Bot

## 项目概述

这是一个基于强化学习的虚拟货币自动交易机器人。它使用深度学习模型来预测市场走势，并根据预测的结果自动进行交易。主要包括以下功能：

- 使用 LSTM 模型对市场数据进行预测。
- 实时获取 Binance 的虚拟货币数据。
- 使用 PPO 算法训练强化学习模型。
- 实时进行交易决策并保存更新后的模型。

## 项目结构
`
crypto_trading_bot/
├── data/
│   └── saved_model.pth        # 模型保存路径
├── train_ppo.py               # 训练 PPO 模型并保存
├── predict_ppo.py             # 加载并使用模型进行预测
├── rl_env.py                  # 强化学习环境定义
├── model.py                   # 模型定义
├── utils.py                   # 数据处理和技术指标工具
├── config.py                  # 配置文件 (Binance API 密钥)
├── real_time_train.py         # 实时训练和模型更新
├── requirements.txt           # 依赖包
└── README.md                  # 项目说明文档
`

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用说明
1.训练模型：运行 train_ppo.py，模型将根据历史数据进行训练并保存在 data/saved_model.pth 中。
```bash
python train_ppo.py
```

2.预测：运行 predict_ppo.py，加载训练好的模型，使用实时数据预测并执行买卖操作。
```bash
python predict_ppo.py
```

3.实时训练与模型更新：运行 real_time_train.py，每小时进行一次模型训练，并保存更新的模型。
```bash
python real_time_train.py
```