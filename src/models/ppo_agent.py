from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.env.portfolio_env import PortfolioEnv

def make_env(env_id, df, seed, rank):
    def _init():
        env = PortfolioEnv(df)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class PPOAgent:
    """Proximal Policy Optimization Agent."""
    def __init__(self, df, seed=42):
        self.df = df
        self.seed = seed
        # Paper specifies n_envs=10
        self.n_envs = 10
        self.envs = DummyVecEnv([make_env("PortfolioEnv-v0", self.df, self.seed, i) for i in range(self.n_envs)])
        # PPO params are set in config
        self.model = PPO(
            config.PPO_PARAMS["policy"],
            self.envs,
            learning_rate=self._linear_schedule(config.PPO_PARAMS["learning_rate"]),
            n_steps=config.PPO_PARAMS["n_steps"],
            batch_size=config.PPO_PARAMS["batch_size"],
            n_epochs=config.PPO_PARAMS["n_epochs"],
            gamma=config.PPO_PARAMS["gamma"],
            gae_lambda=config.PPO_PARAMS["gae_lambda"],
            clip_range=config.PPO_PARAMS["clip_range"],
            policy_kwargs=config.PPO_PARAMS["policy_kwargs"],
            verbose=1,
            seed=self.seed
        )

    def _linear_schedule(self, initial_value):
        """
        Linear learning rate schedule.
        Paper: Linear decay from 3e-4 to 1e-5
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0
            """
            final_value = 1e-5
            return progress_remaining * (initial_value - final_value) + final_value

        return func

    def train(self, total_timesteps=7_500_000, seed_model_path=None):
        if seed_model_path and os.path.exists(seed_model_path):
            print(f"Loading seed policy from {seed_model_path}")
            self.model.set_parameters(seed_model_path)

        print(f"Training PPO agent with seed {self.seed} for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path, env=self.envs)

    def predict(self, state):
        action, _states = self.model.predict(state, deterministic=True)
        return action
