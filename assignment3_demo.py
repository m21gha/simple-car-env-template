# assignment3_demo.py

import time
import gym
import torch
import numpy as np
import simple_driving             # registers 'SimpleDriving-v0'
from training import QNetwork     # your QNetwork class

def demo_model(label, model_path, state_dims, num_episodes=3, render_mode=True):
    print(f"\n=== Demonstrating {label} ({state_dims}-D policy) ===")
    env = gym.make(
        'SimpleDriving-v0',
        apply_api_compatibility=True,
        renders=render_mode,
        isDiscrete=True
    ).unwrapped

    action_size = env.action_space.n
    q_net = QNetwork(state_dims, action_size)
    q_net.load_state_dict(torch.load(model_path, weights_only=True))
    q_net.eval()

    for ep in range(1, num_episodes+1):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        s   = np.array(obs, dtype=np.float32)[:state_dims]
        print(f"Episode {ep} initial observation: {s}")

        total_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                a = q_net(torch.from_numpy(s).unsqueeze(0)).argmax(dim=1).item()

            res = env.step(a)
            if len(res) == 4:
                nxt_obs, reward, done, _ = res
            else:
                nxt_obs, reward, term, trunc, _ = res
                done = term or trunc

            s = np.array(nxt_obs, dtype=np.float32)[:state_dims]
            total_reward += reward

            if render_mode:
                time.sleep(0.01)

        print(f"→ {label} Episode {ep}  Reward: {total_reward:.2f}\n")

    env.close()


def main():
    # group A: just Parts 1 & 2 (no reward‐bonus print)
    models_12 = [
        ("Part 1: Uniform ε-greedy",    "paths/simple_driving_qlearning_part1.pth", 2),
        ("Part 2: Non-uniform ε-greedy","paths/simple_driving_qlearning_part2.pth", 2),
    ]
    # group B: only Part 3 (reward bonus)
    models_3 = [
        ("Part 3: Reward bonus",        "paths/simple_driving_qlearning_part3.pth", 2),
    ]
    # group C: only Part 4 (obstacles, 4‐D)
    models_4 = [
        ("Part 4: Obstacles",           "paths/simple_driving_qlearning_part4_improved.pth", 4),
    ]

    # uncomment the model to see
    #MODELS = models_12     # run only Parts 1 & 2
    #MODELS = models_3      # run only Part 3
    MODELS = models_4      # run only Part 4

    for label, path, dims in MODELS:
        demo_model(
            label,
            model_path=path,
            state_dims=dims,
            num_episodes=3,    # change this for more/less episodes
            render_mode=True   # true means opens the GUI
        )

if __name__ == "__main__":
    main()
