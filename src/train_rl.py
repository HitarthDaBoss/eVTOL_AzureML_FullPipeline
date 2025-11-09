#!/usr/bin/env python3
import os
import sys
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

# ----------------------
# CPU optimization
# ----------------------
device = torch.device("cpu")
torch.set_num_threads(4)  # match your 4 vCPUs
print(f"Using device: {device}, Num threads: {torch.get_num_threads()}")

# ----------------------
# Add repo root for imports
# ----------------------
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.utils import load_config, set_seed, save_checkpoint, load_checkpoint, RunningMeanStd, ensure_dir
from src.env import EVTOLEnv
from src.model import ActorCritic

# ----------------------
# Helpers
# ----------------------
def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - float(dones[t])
        nextvalue = values[t+1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nextvalue * nextnonterminal - values[t]
        lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam
    returns = advantages + np.array(values[:T], dtype=np.float32)
    return advantages, returns

def minibatches_permuted(indices, minibatch_size):
    for start in range(0, len(indices), minibatch_size):
        yield indices[start:start+minibatch_size]

# ----------------------
# PPO Trainer
# ----------------------
class PPOTrainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = torch.device(device)
        self.env = EVTOLEnv(cfg)
        self.obs_dim = int(self.env.observation_space.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])
        self.act_low = torch.tensor(self.env.action_space.low, dtype=torch.float32).to(self.device)
        self.act_high = torch.tensor(self.env.action_space.high, dtype=torch.float32).to(self.device)

        # Model + optimizer
        self.model = ActorCritic(self.obs_dim, self.act_dim, hidden_dim=cfg.get('model', {}).get('hidden_dim', 256)).to(self.device)
        lr = cfg['training']['rl']['lr']
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.8, patience=10, verbose=True)

        # Hyperparameters
        rl = cfg['training']['rl']
        self.gamma = float(rl.get('gamma', 0.99))
        self.lam = float(rl.get('gae_lambda', 0.95))
        self.clip_eps = float(rl.get('clip_eps', 0.2))
        self.ent_coef = float(rl.get('ent_coef', 0.0))
        self.vf_coef = float(rl.get('vf_coef', 0.5))
        self.max_grad_norm = float(rl.get('max_grad_norm', 0.5))
        self.rollout_len = int(rl.get('update_steps', 512))  # CPU-optimized
        self.minibatch_size = int(rl.get('batch_size', 32))   # CPU-optimized
        self.ppo_epochs = int(rl.get('ppo_epochs', 4))        # CPU-optimized
        self.total_epochs = int(rl.get('epochs', 1000))
        self.save_interval = int(rl.get('save_interval', 10)) # CPU-optimized frequent saving
        self.energy_coef = float(rl.get('energy_coef', 0.0))
        self.energy_scale = float(rl.get('energy_scale', 1.0))

        # Logging
        self.log_dir = ensure_dir(cfg.get('log_dir', 'runs/logs'))
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Observation normalization
        self.obs_rms = RunningMeanStd(shape=(self.obs_dim,))

        # Checkpoints
        self.ckpt_dir = ensure_dir(cfg.get('ckpt_dir', 'runs/checkpoints'))

    def scale_action(self, raw_action):
        return raw_action * (self.act_high - self.act_low) / 2.0 + (self.act_high + self.act_low) / 2.0

    # ----------------------
    # Collect rollout
    # ----------------------
    def collect_rollout(self):
        obs_buf, actions_buf, logp_buf, rewards_buf, dones_buf, values_buf = [], [], [], [], [], []
        obs, _ = self.env.reset()
        obs = np.array(obs, dtype=np.float32)

        for step in range(self.rollout_len):
            self.obs_rms.update(obs[None, :])
            obs_norm = self.obs_rms.normalize(obs)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32).to(self.device).unsqueeze(0)

            with torch.no_grad():
                mu, sigma, value = self.model.forward(obs_t)
                mu, sigma, value = mu.squeeze(0), sigma.squeeze(0), value.squeeze(0).item()
                dist = Normal(mu, sigma)
                action = dist.rsample()
                logp = dist.log_prob(action).sum().item()

            scaled_action = self.scale_action(action).cpu().numpy()
            ret = self.env.step(scaled_action)
            if len(ret) == 5:
                next_obs, reward, terminated, truncated, info = ret
                done = terminated or truncated
            else:
                next_obs, reward, done, info = ret

            if self.energy_coef != 0.0:
                energy_pen = self.energy_coef * (np.sum(np.square(scaled_action)) * self.energy_scale)
                reward = reward - energy_pen

            obs_buf.append(obs_norm)
            actions_buf.append(action.cpu().numpy())
            logp_buf.append(logp)
            rewards_buf.append(float(reward))
            dones_buf.append(bool(done))
            values_buf.append(value)

            obs = np.array(next_obs, dtype=np.float32)
            if done:
                obs, _ = self.env.reset()
                obs = np.array(obs, dtype=np.float32)

        obs_norm = self.obs_rms.normalize(obs)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = self.model.forward(obs_t)
            last_value = last_value.squeeze(0).item()
        values_buf.append(last_value)

        return {
            'obs': np.array(obs_buf, dtype=np.float32),
            'actions': np.array(actions_buf, dtype=np.float32),
            'logp': np.array(logp_buf, dtype=np.float32),
            'rewards': np.array(rewards_buf, dtype=np.float32),
            'dones': np.array(dones_buf, dtype=np.bool_),
            'values': np.array(values_buf, dtype=np.float32),
        }

    # ----------------------
    # Update
    # ----------------------
    def update(self, batch):
        advs, returns = compute_gae(batch['rewards'].tolist(), batch['values'].tolist(), batch['dones'].tolist(), self.gamma, self.lam)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        obs = torch.tensor(batch['obs'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.float32).to(self.device)
        old_logp = torch.tensor(batch['logp'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advs = torch.tensor(advs, dtype=torch.float32).to(self.device)

        N = obs.shape[0]
        indices = np.arange(N)
        policy_losses, value_losses, entropies = [], [], []

        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for mb_inds in minibatches_permuted(indices, self.minibatch_size):
                mb_obs = obs[mb_inds]
                mb_actions = actions[mb_inds]
                mb_oldlogp = old_logp[mb_inds]
                mb_returns = returns[mb_inds]
                mb_advs = advs[mb_inds]

                mu, sigma, value = self.model.forward(mb_obs)
                dist = Normal(mu, sigma)
                new_logp = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                ratio = torch.exp(new_logp - mb_oldlogp)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - value.squeeze(-1)).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return np.mean(policy_losses), np.mean(value_losses), np.mean(entropies)

    # ----------------------
    # Train
    # ----------------------
    def train(self, resume_path=None):
        start_epoch = 0
        if resume_path:
            ckpt = load_checkpoint(resume_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            print("[trainer] Resumed from", resume_path)

        for epoch in range(start_epoch, self.total_epochs):
            t0 = time.time()
            batch = self.collect_rollout()
            pol_loss, val_loss, entropy = self.update(batch)
            rollout_reward = float(batch['rewards'].sum())
            self.scheduler.step(rollout_reward)

            self.writer.add_scalar('train/rollout_reward', rollout_reward, epoch)
            self.writer.add_scalar('train/policy_loss', pol_loss, epoch)
            self.writer.add_scalar('train/value_loss', val_loss, epoch)
            self.writer.add_scalar('train/entropy', entropy, epoch)
            self.writer.add_scalar('train/obs_mean', np.mean(self.obs_rms.mean), epoch)

            if epoch % self.save_interval == 0 or epoch == self.total_epochs - 1:
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'obs_rms': {
                        'mean': self.obs_rms.mean.tolist(),
                        'var': self.obs_rms.var.tolist(),
                        'count': float(self.obs_rms.count)
                    },
                    'cfg': self.cfg
                }
                save_checkpoint(ckpt, os.path.join(self.ckpt_dir, f"checkpoint_epoch_{epoch}.pth"))

            print(f"[Epoch {epoch}] reward={rollout_reward:.2f} pol_loss={pol_loss:.4f} val_loss={val_loss:.4f} entropy={entropy:.4f} time={time.time()-t0:.2f}s")

        self.writer.close()
        print("Training complete")

# ----------------------
# CLI
# ----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', type=str, default='configs/default.yaml')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=str, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.cfg)
    set_seed(args.seed)
    trainer = PPOTrainer(cfg, device=args.device)
    trainer.train(resume_path=args.resume)

if __name__ == '__main__':
    main()
