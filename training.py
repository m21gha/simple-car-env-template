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

def select_action(state, q_net, action_size, epsilon):
    if np.random.rand() < epsilon:
        # non-uniform exploration biasing forward
        probs = np.array([
            0.05, 0.05, 0.05,
            0.10, 0.10, 0.10,
            0.20, 0.20, 0.15,
        ], dtype=np.float32)
        return int(np.random.choice(np.arange(action_size), p=probs))
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = q_net(state_tensor)
        return torch.argmax(q_vals, dim=1).item()

def replay(replay_memory, q_net, target_net, optimizer, criterion, gamma, batch_size):
    if len(replay_memory) < batch_size:
        return
    minibatch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)

    states      = torch.FloatTensor(np.stack(states))
    actions     = torch.LongTensor(actions).unsqueeze(1)
    rewards     = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.stack(next_states))
    dones       = torch.FloatTensor(dones)

    q_values = q_net(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
    q_targets = rewards + gamma * next_q_values * (1 - dones)

    loss = criterion(q_values, q_targets)
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()

def main():
    # Hyperparameters
    num_episodes   = 2000
    max_steps      = 200
    gamma          = 0.999
    epsilon        = 1.0
    epsilon_min    = 0.01
    epsilon_decay  = np.exp(np.log(epsilon_min) / num_episodes)
    batch_size     = 64
    target_update  = 50   # episodes between target network sync

    # Environment setup
    env = gym.make('SimpleDriving-v0',
                   apply_api_compatibility=True,
                   renders=False,
                   isDiscrete=True).unwrapped
    state, _    = env.reset()
    state       = np.array(state, dtype=np.float32).flatten()
    state_size  = len(state)
    action_size = env.action_space.n

    # Q-network and target network
    q_net      = QNetwork(state_size, action_size)
    target_net = QNetwork(state_size, action_size)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer      = optim.Adam(q_net.parameters(), lr=0.001)
    criterion      = nn.MSELoss()
    replay_memory  = deque(maxlen=10000)
    running_rewards = deque(maxlen=100)

    print("Starting training...")
    for episode in range(1, num_episodes+1):
        state, _ = env.reset()
        state    = np.array(state, dtype=np.float32).flatten()
        total_reward = 0.0

        for step in range(max_steps):
            action = select_action(state, q_net, action_size, epsilon)
            next_state, reward, done, *_ = env.step(action)
            next_state = np.array(next_state, dtype=np.float32).flatten()

            total_reward += reward
            replay_memory.append((state, action, reward, next_state, float(done)))
            state = next_state

            replay(replay_memory, q_net, target_net, optimizer, criterion, gamma, batch_size)
            if done:
                break

        # sync target network
        if episode % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())

        # epsilon decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # log
        running_rewards.append(total_reward)
        if episode % 100 == 0:
            avg_reward = sum(running_rewards)/len(running_rewards)
            print(f"Ep {episode}/{num_episodes}  Avg100: {avg_reward:.2f}  ε: {epsilon:.3f}")

    # Save the model
    torch.save(q_net.state_dict(), "paths/simple_driving_qlearning_part4_improved.pth")
    print("Training done. Model saved!")

if __name__ == "__main__":
    main()