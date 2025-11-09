import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.env import EVTOLEnv
from src.model import ActorCritic
from src.utils import load_config

def load_model(path, obs_dim, act_dim, device='cpu'):
    model = ActorCritic(obs_dim, act_dim)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    model.to(device).eval()
    return model

def rollout(env, model=None, policy_fn=None, episodes=5, device='cpu'):
    results = []
    for ep in range(episodes):
        obs,_ = env.reset()
        traj = []
        total = 0.0
        done = False
        while not done:
            if model:
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    mu, sigma, _ = model.forward(obs_t)
                    action = mu.squeeze(0).cpu().numpy()
            elif policy_fn:
                action = policy_fn(obs)
            else:
                action = np.zeros(env.action_space.shape)
            obs, reward, done, info = env.step(action)
            traj.append(obs[:3])
            total += reward
        results.append((np.array(traj), total))
    return results

def plot_trajectories(results, title="Trajectories"):
    plt.figure(figsize=(8,6))
    for traj, total in results:
        plt.plot(traj[:,0], traj[:,1], '-')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    env = EVTOLEnv(cfg)
    # load model optionally
    # model = load_model("runs/checkpoints/checkpoint_epoch_50.pth", env.observation_space.shape[0], env.action_space.shape[0])
    model = None
    res = rollout(env, model=model, episodes=3)
    plot_trajectories(res)
    for i,(traj,total) in enumerate(res):
        print(f"Episode {i} reward: {total:.2f}")
