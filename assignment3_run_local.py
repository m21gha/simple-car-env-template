import gym
import simple_driving  # This registers your custom 'SimpleDriving-v0' environment
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

print("Starting training...")

# Create and unwrap the environment
env = gym.make('SimpleDriving-v0', apply_api_compatibility=True, renders=False, isDiscrete=True)
env = env.unwrapped
state, info = env.reset()
print("Observation:", state)  # Should print a 4-element vector

# Reset the environment using the new Gym API (which returns a tuple)
state, info = env.reset()
# Ensure the state is a flat NumPy array
state = np.array(state, dtype=np.float32).flatten()
state_size = len(state)  # Expecting, for example, 2 elements
action_size = env.action_space.n  # Typically 9 discrete actions

# Define a simple Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialize network, optimizer, and loss function
q_network = QNetwork(state_size, action_size)
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Hyperparameters and replay memory
num_episodes = 500
max_steps = 200
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
replay_memory = deque(maxlen=10000)

# PART 1 (COMMENT OUT WHEN NOT IN USE)
#def select_action(state, epsilon):
#    if np.random.rand() < epsilon:
#        # uniform exploration
#        return np.random.randint(action_size)
#    else:
#        # greedy exploitation
#        state_tensor = torch.FloatTensor(state).unsqueeze(0)
#        with torch.no_grad():
#            q_values = q_network(state_tensor)
#        return torch.argmax(q_values, dim=1).item()

# PART 2
def select_action(state, epsilon):
    if np.random.rand() <= epsilon:
        # Define a non-uniform distribution for the 9 actions.
        # Example:
        # 0: Reverse-Left    -> 0.05
        # 1: Reverse         -> 0.05
        # 2: Reverse-Right   -> 0.05
        # 3: Steer-Left      -> 0.10
        # 4: No throttle     -> 0.10
        # 5: Steer-Right     -> 0.10
        # 6: Forward-right   -> 0.20
        # 7: Forward         -> 0.20
        # 8: Forward-left    -> 0.15
        probs = np.array([0.05, 0.05, 0.05, 0.10, 0.10, 0.10, 0.20, 0.20, 0.15])
        return int(np.random.choice(np.arange(action_size), p=probs))
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state_tensor)
        return torch.argmax(q_values).item()
    
# Experience replay function using np.stack for uniform shape
def replay():
    if len(replay_memory) < batch_size:
        return
    minibatch = random.sample(replay_memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*minibatch)
    
    # Use np.stack to build a consistent array
    states = torch.FloatTensor(np.stack(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.stack(next_states))
    dones = torch.FloatTensor(dones)
    
    q_values = q_network(states).gather(1, actions).squeeze(1)
    with torch.no_grad():
        next_q_values = q_network(next_states).max(1)[0]
    q_targets = rewards + gamma * next_q_values * (1 - dones)
    
    loss = criterion(q_values, q_targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main training loop
for episode in range(num_episodes):
    state, info = env.reset()  # Unpack state and info from reset
    state = np.array(state, dtype=np.float32).flatten()  # Ensure flat array

    # <-- print initial observation each episode
    print(f"Episode {episode:3d} start obs:", state)
    
    total_reward = 0
    for step in range(max_steps):
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32).flatten()  # Ensure flat array
        total_reward += reward
        
        # Store transition ensuring state consistency
        replay_memory.append((state, action, reward, next_state, float(done)))
        state = next_state
        
        replay()  # Train on a mini-batch
        
        if done:
            break
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    
    print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

# Save the trained model weights
torch.save(q_network.state_dict(), "simple_driving_qlearning_part4.pth")
print("Model saved!")
