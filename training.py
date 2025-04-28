import gym
import simple_driving               # registers 'SimpleDriving-v0'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

#### PART 1
def select_action(state, q_net, action_size, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = q_net(state_tensor)
        return torch.argmax(q_vals, dim=1).item()

def replay(replay_memory, q_net, optimizer, criterion, gamma, batch_size):
    if len(replay_memory) < batch_size:
        return
    minibatch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states      = torch.FloatTensor(np.stack(states))
    actions     = torch.LongTensor(actions).unsqueeze(1)
    rewards     = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.stack(next_states))
    dones       = torch.FloatTensor(dones)

    q_values      = q_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = q_net(next_states).max(1)[0]
    q_targets     = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(q_values, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def main():
    # Hyperparameters
    num_episodes  = 500
    max_steps     = 200
    gamma         = 0.99
    epsilon       = 1.0
    epsilon_min   = 0.01
    epsilon_decay = 0.995
    batch_size    = 64

    # Environment setup
    env = gym.make('SimpleDriving-v0',
                   apply_api_compatibility=True,
                   renders=False,
                   isDiscrete=True).unwrapped
    state, _   = env.reset()
    state      = np.array(state, dtype=np.float32).flatten()
    state_size = len(state)
    action_size= env.action_space.n

    # Q-network, optimizer, loss
    q_net     = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_memory = deque(maxlen=10000)

    print("Starting training...")
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        state    = np.array(state, dtype=np.float32).flatten()
        total_reward = 0.0

        for step in range(max_steps):
            action = select_action(state, q_net, action_size, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()

            total_reward += reward
            replay_memory.append((state, action, reward, next_state, float(done)))
            state = next_state

            replay(replay_memory, q_net, optimizer, criterion, gamma, batch_size)
            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode}/{num_episodes}  Total Reward: {total_reward:.2f}  Epsilon: {epsilon:.3f}")

    # Adjust filename if you saved different parts separately
    torch.save(q_net.state_dict(), "paths/simple_driving_qlearning_part4.pth")
    print("Model saved!")

if __name__ == "__main__":
    main()
