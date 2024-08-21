# 强化学习环境
import numpy as np
import gym
from gym import spaces
from utils import fetch_ohlcv, compute_indicators


class TradingEnv(gym.Env):
    """自定义的强化学习交易环境"""
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol='BTC/USDT', window_size=50):
        super(TradingEnv, self).__init__()

        self.symbol = symbol
        self.window_size = window_size
        self.data = None
        self.current_step = 0
        self.balance = 10000  # 初始资产
        self.position = 0  # 持仓状态 (+1: 买入, -1: 卖出)
        self.entry_price = 0  # 记录买入或卖出的价格

        # 状态空间：包含K线数据和技术指标
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 6), dtype=np.float32)
        # 动作空间：买入、卖出、持仓
        self.action_space = spaces.Discrete(3)

    def reset(self):
        """重置环境"""
        self.data = compute_indicators(fetch_ohlcv(self.symbol, '1m', 500))
        self.current_step = self.window_size
        self.balance = 10000
        self.position = 0
        self.entry_price = 0
        return self._get_observation()

    def _get_observation(self):
        """返回当前的市场状态"""
        return self.data.iloc[self.current_step - self.window_size:self.current_step].values

    def step(self, action):
        """执行动作，返回下一个状态、奖励和是否结束"""
        current_price = self.data['close'].iloc[self.current_step]
        reward = 0

        if action == 1:  # 买入
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:
                reward = (self.entry_price - current_price) * 100
                self.balance += reward
                self.position = 0
                self.entry_price = 0

        elif action == 2:  # 卖出
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:
                reward = (current_price - self.entry_price) * 100
                self.balance += reward
                self.position = 0
                self.entry_price = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        reward = reward / 100  # 奖励与损益成比例

        return self._get_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")
