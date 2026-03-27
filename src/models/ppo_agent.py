import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv   # FIX: was DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.env.portfolio_env import PortfolioEnv


def make_env(df, seed: int, rank: int):
    """Factory function for a single vectorised sub-environment."""
    def _init():
        env = PortfolioEnv(df)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class PPOAgent:
    """
    Proximal Policy Optimisation agent wrapping StableBaselines3.

    Fixes vs original
    -----------------
    1. SubprocVecEnv (true multiprocessing) instead of DummyVecEnv.
       The paper explicitly uses SubprocVecEnv for parallel rollout collection
       across n_envs=10 independent environment instances (Section 5.2).

    2. log_std_init=-1 is now set via config.PPO_PARAMS["policy_kwargs"].
       The paper states: "initialise the policy with log_std_init = −1"
       (Section 5.2). This tightens the initial action distribution.
    """

    def __init__(self, df, seed: int = 42):
        self.df     = df
        self.seed   = seed
        self.n_envs = config.N_ENVS   # 10

        # FIX: SubprocVecEnv — true multiprocessing (paper Section 5.2)
        #      DummyVecEnv ran all envs sequentially; no parallelism benefit.
        self.envs = SubprocVecEnv(
            [make_env(self.df, self.seed, i) for i in range(self.n_envs)],
            start_method="fork",       # "fork" is fastest on Linux/macOS
        )

        self.model = PPO(
            config.PPO_PARAMS["policy"],
            self.envs,
            learning_rate  = self._linear_schedule(config.PPO_PARAMS["learning_rate"]),
            n_steps        = config.PPO_PARAMS["n_steps"],
            batch_size     = config.PPO_PARAMS["batch_size"],
            n_epochs       = config.PPO_PARAMS["n_epochs"],
            gamma          = config.PPO_PARAMS["gamma"],
            gae_lambda     = config.PPO_PARAMS["gae_lambda"],
            clip_range     = config.PPO_PARAMS["clip_range"],
            policy_kwargs  = config.PPO_PARAMS["policy_kwargs"],  # includes log_std_init=-1
            verbose        = 1,
            seed           = self.seed,
        )

    def _linear_schedule(self, initial_value: float):
        """
        Linear LR decay from initial_value (3e-4) to final_value (1e-5),
        as a function of progress_remaining (1 → 0).
        Paper: "learning rate decaying from 3e-4, annealed to a final value of 1e-5"
        """
        final_value = 1e-5

        def func(progress_remaining: float) -> float:
            return progress_remaining * (initial_value - final_value) + final_value

        return func

    def train(self, total_timesteps: int = 7_500_000, seed_model_path=None):
        """
        Train the PPO agent.

        seed_model_path: if provided, initialise this agent's parameters from
                         the best agent of the previous window (warm-start).
                         Paper Section 5.2: "This agent is used as a seed policy
                         for the next group of 5 agents."
        """
        if seed_model_path and os.path.exists(str(seed_model_path)):
            print(f"  Warm-starting from {seed_model_path}")
            self.model.set_parameters(str(seed_model_path))

        print(f"  Training PPO (seed={self.seed}) for {total_timesteps:,} timesteps ...")
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path):
        self.model.save(str(path))

    def load(self, path):
        self.model = PPO.load(str(path), env=self.envs)

    def predict(self, state, deterministic: bool = True):
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action            gae_lambda=config.PPO_PARAMS["gae_lambda"],
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
