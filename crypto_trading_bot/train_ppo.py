import torch
import torch.optim as optim
from torch.distributions import Categorical
from rl_env import TradingEnv
from model import LSTMModel

class PPOAgent:
    def __init__(self, model, lr=1e-3, gamma=0.99, eps_clip=0.2):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.model(state)
        dist = Categorical(logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, rewards, log_probs, states, actions):
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss = -log_prob * reward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

def train_ppo():
    env = TradingEnv()
    model = LSTMModel(input_size=6, hidden_size=64, output_size=3)
    agent = PPOAgent(model)

    for episode in range(1000):
        state = env.reset()
        rewards = []
        log_probs = []
        done = False

        while not done:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        agent.update(rewards, log_probs, [state], actions)
        print(f'Episode {episode}: Reward: {sum(rewards)}')

    # 保存模型
    torch.save(model.state_dict(), 'data/saved_model.pth')

if __name__ == '__main__':
    train_ppo()