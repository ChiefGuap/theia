#!/usr/bin/env python3
"""
train_ppo_theia.py

Train PPO on FetchReach-v4 by wrapping the env to emit Theia CLS embeddings
as observations, then training a simple MLP policy. Embedding size is
automatically inferred so you never get dimension mismatches.
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics            # registers FetchReach-v4
import torch
from torchvision import transforms
from transformers import AutoModel
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import spaces
from PIL import Image

class TheiaEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, model_id="theaiinstitute/theia-small-patch16-224-cdiv"):
        super().__init__(env)
        # Load Theia model once
        self.theia = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval()
        # Pre‑processing: PIL.Image → resize → Tensor [0,1]
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),  # expects PIL.Image
            transforms.ToTensor(),          # outputs FloatTensor [0,1]
        ])
        # Infer embedding dim by embedding one frame
        self.env.reset()
        frame = self.env.render()
        emb = self._embed(frame)
        d = emb.shape[-1]  # actual embedding size
        # Set the new observation space to (d,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(d,), dtype=np.float32
        )
        self.action_space = env.action_space

    def _embed(self, frame: np.ndarray) -> np.ndarray:
        # frame: H×W×C uint8
        img = Image.fromarray(frame)              # PIL.Image
        t   = self.tf(img).unsqueeze(0)           # [1,3,224,224]
        with torch.no_grad():
            out = self.theia(x=t)
        # Get token embeddings
        if isinstance(out, dict):
            tokens = next(iter(out.values()))
        elif isinstance(out, (tuple, list)):
            tokens = out[0]
        else:
            tokens = out
        # tokens: [1, N, D]
        cls = tokens[:, 0, :].cpu().numpy()       # (1, D)
        return cls[0]                             # (D,)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        frame = self.env.render()
        return self._embed(frame), info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        return self._embed(frame), reward, terminated, truncated, info

def make_env():
    base = gym.make("FetchReach-v4", render_mode="rgb_array")
    wrapped = TheiaEmbeddingWrapper(base)
    return Monitor(wrapped)

def main():
    venv = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        tensorboard_log="./ppo_theia_tensorboard/"
    )

    model.learn(total_timesteps=200_000)
    model.save("ppo_fetchreach_theia")
    print("✅ Done — model saved as ppo_fetchreach_theia.zip")

    # Optional evaluation
    obs, _ = venv.reset()
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = venv.step(action)
        if terminated or truncated:
            obs, _ = venv.reset()

if __name__ == "__main__":
    main()
