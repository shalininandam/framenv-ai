import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class FarmNet(nn.Module):
    def __init__(self):
        super(FarmNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.network(x)

class FarmAgent:
    def __init__(self):
        self.model = FarmNet()
        self.memory = deque(maxlen=1000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        self.criterion = nn.MSELoss()
        self.action_names = [
            "Water Crops",
            "Apply Fertilizer",
            "Apply Pesticide",
            "Do Nothing",
            "Harvest"
        ]

    def state_to_tensor(self, state):
        values = [
            state["day"] / 30.0,
            state["water_level"] / 100.0,
            state["soil_health"] / 100.0,
            state["pest_level"] / 100.0,
            state["growth_stage"] / 5.0,
            state["yield_score"] / 200.0,
            ["sunny","cloudy","rainy","stormy"].index(
                state["weather"]) / 3.0
        ]
        return torch.FloatTensor(values).unsqueeze(0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        with torch.no_grad():
            tensor = self.state_to_tensor(state)
            q_values = self.model(tensor)
            return q_values.argmax().item()

    def remember(self, state, action, reward,
                 next_state, done):
        self.memory.append((state, action, reward,
                            next_state, done))

    def learn(self):
        if len(self.memory) < 32:
            return 0

        batch = random.sample(self.memory, 32)
        total_loss = 0

        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_tensor = self.state_to_tensor(next_state)
                with torch.no_grad():
                    target = reward + self.gamma * \
                        self.model(next_tensor).max().item()

            tensor = self.state_to_tensor(state)
            current_q = self.model(tensor)
            target_q = current_q.clone().detach()
            target_q[0][action] = target

            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / 32

    def run_episode(self, env):
        state = env.reset()
        total_reward = 0
        steps = []
        done = False

        while not done:
            action = self.choose_action(state)
            next_state, reward, done, info = env.step(action)
            self.remember(state, action, reward,
                         next_state, done)
            loss = self.learn()
            
            steps.append({
                "day": state["day"],
                "action": self.action_names[action],
                "reward": round(reward, 2),
                "water": round(state["water_level"], 1),
                "soil": round(state["soil_health"], 1),
                "pest": round(state["pest_level"], 1),
                "growth": round(state["growth_stage"], 2),
                "weather": state["weather"],
                "info": info
            })

            total_reward += reward
            state = next_state

        return round(total_reward, 2), steps