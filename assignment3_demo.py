# assignment3_demo.py

import time
import gym
import torch
import numpy as np
import simple_driving             # registers 'SimpleDriving-v0'
from training import QNetwork     # your QNetwork class from training.py

def demo_model(label, model_path, state_dims, num_episodes=3, render_mode=True):
    print(f"\n=== Demonstrating {label} ({state_dims}-D policy) ===")

    # create the env with GUI turned on/off
    env = gym.make(
        'SimpleDriving-v0',
        apply_api_compatibility=True,
        renders=render_mode,    # True → open PyBullet GUI
        isDiscrete=True
    ).unwrapped

    # load the correct network
    action_size = env.action_space.n
    q_net = QNetwork(state_dims, action_size)
    q_net.load_state_dict(torch.load(model_path, weights_only=True))
    q_net.eval()

    for ep in range(1, num_episodes+1):
        # reset gives (obs, info)
        obs, _ = env.reset()
        s = np.array(obs, dtype=np.float32)[:state_dims]
        total_reward = 0.0
        done = False

        while not done:
            # pick greedy action
            with torch.no_grad():
                a = q_net(torch.from_numpy(s).unsqueeze(0)).argmax(dim=1).item()

            # step() may return 4 or 5 items
            result = env.step(a)
            if len(result) == 4:
                nxt_obs, reward, done, _ = result
            else:
                nxt_obs, reward, terminated, truncated, _ = result
                done = terminated or truncated

            nxt_obs = np.array(nxt_obs, dtype=np.float32)
            s = nxt_obs[:state_dims]
            total_reward += reward

            # The GUI window is already active; just pause so you can watch
            if render_mode:
                time.sleep(0.01)

        print(f"→ {label} Episode {ep}  Reward: {total_reward:.2f}")

    env.close()


def main():
    models = [
        ("Part 1: Uniform ε-greedy",     "paths/simple_driving_qlearning_part1.pth", 2),
        ("Part 2: Non-uniform ε-greedy", "paths/simple_driving_qlearning_part2.pth", 2),
        ("Part 3: Reward bonus",         "paths/simple_driving_qlearning_part3.pth", 2),
        ("Part 4: Obstacles",            "paths/simple_driving_qlearning_part4.pth", 4),
    ]

    for label, path, dims in models:
        demo_model(
            label,
            model_path=path,
            state_dims=dims,
            num_episodes=3,    # change to however many runs you like
            render_mode=True   # True opens the PyBullet GUI
        )

if __name__ == "__main__":
    main()
