import torch
import time
from utils import fetch_ohlcv, compute_indicators, load_model
from model import LSTMModel
from rl_env import TradingEnv


def predict():
    model = load_model(LSTMModel)
    env = TradingEnv()
    state = env.reset()

    while True:
        with torch.no_grad():
            logits = model(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(logits).item()
            if action == 0:
                print("Action: Hold")
            elif action == 1:
                print("Action: Buy")
            else:
                print("Action: Sell")

            next_state, reward, done, _ = env.step(action)
            state = next_state

            if done:
                state = env.reset()

            time.sleep(60)  # 每分钟预测一次


if __name__ == '__main__':
    predict()
