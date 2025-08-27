"""
ai_module.py
-------------

This module provides AI functionality for the trading bot, including a
reinforcement learning (RL) environment, deep Q-learning agent, genetic
optimization, and integration with a local large language model (LLM).

To ensure the bot runs even when certain heavy dependencies are missing,
imports of optional libraries such as PyTorch, gymnasium, stable_baselines3
and sklearn_genetic are wrapped in try/except blocks. If these libraries
are unavailable, the RL and genetic optimization functionality will be
disabled gracefully and default fallbacks will be used.
"""

import numpy as np
import pandas as pd
import random
import collections
from pathlib import Path
from config import LLM_MODEL_PATH, RL_ENV_PARAMS
from backtester import backtest
from strategy_modifier import add_new_strategy

import logging
logger = logging.getLogger(__name__)

# Attempt to import TA-Lib; set to None if unavailable
try:
    import talib  # Ensure talib is imported for temp_strategy
except Exception as e:
    logger.warning(f"TA-Lib is not available: {e}")
    talib = None

# Attempt to import heavy optional dependencies. Set to None if unavailable.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    logger.warning(f"Torch is not available: {e}")
    torch = None
    nn = None
    F = None

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    logger.warning(f"Gymnasium is not available: {e}")
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
except Exception as e:
    logger.warning(f"stable_baselines3 PPO is not available: {e}")
    PPO = None

try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer
except Exception as e:
    logger.warning(f"sklearn_genetic is not available: {e}")
    GASearchCV = None
    Integer = None

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception as e:
    logger.warning(f"sklearn RandomForestClassifier is not available: {e}")
    RandomForestClassifier = None

try:
    from llama_cpp import Llama
except Exception as e:
    logger.warning(f"llama_cpp Llama is not available: {e}")
    Llama = None

import logging; logging.getLogger("llama_cpp").setLevel(logging.WARNING)

if gym is not None:
    class TradingEnv(gym.Env):
        """
        Gymnasium environment for the trading bot. The observation consists of a
        fixed number of normalized indicators. The action space is discrete with
        three actions: buy, hold, sell.
        """
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)
            self.current_step = 0
            self.action_space = spaces.Discrete(RL_ENV_PARAMS['action_size'])
            self.observation_space = spaces.Box(low=0, high=1, shape=(RL_ENV_PARAMS['state_size'],), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            self.current_step = 0
            return self._get_obs(), {}

        def step(self, action):
            reward = self._calculate_reward(action)
            self.current_step += 1
            done = self.current_step >= len(self.df) - 1
            truncated = False
            return self._get_obs(), reward, done, truncated, {}

        def _get_obs(self):
            if self.current_step >= len(self.df):
                return np.zeros(RL_ENV_PARAMS['state_size'])
            row = self.df.iloc[self.current_step]
            # Normalize a couple of indicator values; fallback to defaults if missing
            obs = np.array([
                (row.get('TA_RSI_14', 50) / 100),
                (row.get('TA_MACD', 0) + 1) / 2,
            ] + [0] * (RL_ENV_PARAMS['state_size'] - 2))
            return np.clip(obs, 0, 1)

        def _calculate_reward(self, action):
            # Reward based on next period's return
            if self.current_step + 1 >= len(self.df):
                return 0
            next_return = self.df['Close'].pct_change().shift(-1).iloc[self.current_step]
            if action == 0:  # Buy
                return next_return if next_return > 0 else next_return * 0.5
            elif action == 2:  # Sell
                return -next_return if next_return < 0 else -next_return * 0.5
            return 0  # Hold
else:
    # Fallback dummy environment if gymnasium is not available
    class TradingEnv:
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)
            self.current_step = 0
            self.action_space = None
            self.observation_space = None

        def reset(self, *, seed=None, options=None):
            self.current_step = 0
            return np.zeros(RL_ENV_PARAMS['state_size']), {}

        def step(self, action):
            # Always return zero reward and done at end
            self.current_step += 1
            done = self.current_step >= len(self.df) - 1
            truncated = False
            return np.zeros(RL_ENV_PARAMS['state_size']), 0, done, truncated, {}

if torch is not None:
    class DQN(nn.Module):
        """Simple feedforward neural network for DQN agent"""
        def __init__(self, state_size, action_size):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(state_size, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, action_size)

        def forward(self, state):
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
else:
    class DQN:
        def __init__(self, state_size, action_size):
            pass
        def forward(self, state):
            return np.zeros(action_size)

class TradingDQN:
    """
    Deep Q-learning agent for trading decisions.
    """
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lr=0.001, memory_size=2000, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = DQN(state_size, action_size) if torch is not None else None
        if self.model:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if self.model is None:
            return 0
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return np.argmax(q_values.detach().numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size or self.model is None:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class AIModule:
    def __init__(self):
        model_path = Path(LLM_MODEL_PATH) if LLM_MODEL_PATH else None
        if Llama and model_path and model_path.is_file():
            self.llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=512, n_batch=512)
        else:
            if not (model_path and model_path.is_file()):
                logger.warning("LLM model path not set; proceeding without local LLM")
            self.llm = None
        self.rl_model = None
        self.dqn = None
        self.genetic_optimizer = None

    def init_rl_model(self, env):
        if PPO is not None:
            self.rl_model = PPO("MlpPolicy", env, verbose=0)
        else:
            logger.warning("PPO not available for RL model initialization.")

    def init_dqn(self, state_size, action_size):
        self.dqn = TradingDQN(state_size, action_size) if torch is not None else None

    def genetic_optimize(self, X, y, n_estimators_range=(10, 100), max_depth_range=(3, 10)):
        if RandomForestClassifier is None or GASearchCV is None or Integer is None:
            raise RuntimeError("Genetic algorithm dependencies unavailable")
        param_grid = {
            'n_estimators': Integer(n_estimators_range[0], n_estimators_range[1]),
            'max_depth': Integer(max_depth_range[0], max_depth_range[1])
        }
        clf = RandomForestClassifier()
        evolved_estimator = GASearchCV(estimator=clf, param_grid=param_grid, scoring="accuracy", n_jobs=-1)
        evolved_estimator.fit(X, y)
        return evolved_estimator.best_params_

    def llm_decision_support(self, prompt):
        if self.llm is None:
            return "LLM not available."
        try:
            response = self.llm(prompt, max_tokens=200)  # Increased for fuller response
            return response['choices'][0]['text']
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return "LLM call failed."

    def evolve_strategies(self, historical_df, population_size=50, generations=10, mutation_rate=0.1):
        """
        Evolve strategy parameters using a genetic algorithm. On success, writes
        a new strategy function to strategies.py using strategy_modifier.add_new_strategy.
        """
        if talib is None:
            logger.warning("TA-Lib is required for evolve_strategies but is not installed.")
            return None

        gene_ranges = {'fast_period': (5, 20), 'slow_period': (20, 40), 'signal_period': (5, 15)}
        population = self._initialize_population(population_size, gene_ranges)
        for gen in range(generations):
            fitness = []
            for ind in population:
                try:
                    fitness.append(self._calculate_fitness(ind, historical_df))
                except Exception as e:
                    logger.warning(f"Fitness calculation failed: {e}")
                    fitness.append(None)
            # Generate next generation
            new_population = []
            for _ in range(population_size // 2):
                parent1, parent2 = self._selection(population, fitness)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1, mutation_rate, gene_ranges)
                child2 = self._mutate(child2, mutation_rate, gene_ranges)
                new_population.extend([child1, child2])
            population = new_population
        # Select best individual
        best_fitness = [f if f is not None else -np.inf for f in fitness]
        best_index = int(np.argmax(best_fitness))
        best_ind = population[best_index]
        func_name = f"evolved_macd_{random.randint(1, 1000)}"
        strategy_code = (
            f"def {func_name}(df, multi_data=None, sym=None, news_bias=0):\n"
            f"    macd, signal, hist = talib.MACD(df['Close'], fastperiod={best_ind['fast_period']},"
            f" slowperiod={best_ind['slow_period']}, signalperiod={best_ind['signal_period']})\n"
            f"    return np.where(macd > signal, 1, np.where(macd < signal, -1, 0))\n"
        )
        # Write the new strategy to strategies.py
        try:
            add_new_strategy('strategies.py', func_name, strategy_code)
        except Exception as e:
            logger.warning(f"Failed to add evolved strategy: {e}")
        return best_ind

    def _initialize_population(self, size, ranges):
        return [{'fast_period': random.randint(ranges['fast_period'][0], ranges['fast_period'][1]),
                 'slow_period': random.randint(ranges['slow_period'][0], ranges['slow_period'][1]),
                 'signal_period': random.randint(ranges['signal_period'][0], ranges['signal_period'][1])} for _ in range(size)]

    def _calculate_fitness(self, ind, df):
        if talib is None:
            logger.warning("TA-Lib is required for _calculate_fitness but is not installed.")
            return 0

        # Use a temporary MACD strategy with parameters from the individual
        def temp_strategy(inner_df):
            macd, signal, hist = talib.MACD(inner_df['Close'], fastperiod=ind['fast_period'], slowperiod=ind['slow_period'], signalperiod=ind['signal_period'])
            return pd.Series(np.where(macd > signal, 1, np.where(macd < signal, -1, 0)), index=inner_df.index)
        result = backtest(df, 'temp', temp_strategy=temp_strategy)
        return result['sharpe'] if result['sharpe'] else 0

    def _selection(self, population, fitness):
        total = sum(f for f in fitness if f is not None and f > 0)
        if total == 0 or any(f is None for f in fitness):
            return random.choice(population), random.choice(population)
        return random.choices(population, weights=[f / total for f in fitness])[0], \
               random.choices(population, weights=[f / total for f in fitness])[0]

    def _crossover(self, p1, p2):
        # Simple one-point crossover
        return {
            'fast_period': p1['fast_period'],
            'slow_period': p2['slow_period'],
            'signal_period': p1['signal_period']
        }, {
            'fast_period': p2['fast_period'],
            'slow_period': p1['slow_period'],
            'signal_period': p2['signal_period']
        }

    def _mutate(self, ind, rate, ranges):
        for key in ind:
            if random.random() < rate:
                ind[key] = random.randint(ranges[key][0], ranges[key][1])
        return ind

    def train_dqn(self, env, episodes=100, batch_size=32):
        """Train the DQN agent using experience replay."""
        if self.dqn is None:
            logger.warning("DQN not initialized. Call init_dqn first.")
            return
        for e in range(episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.dqn.act(state)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                self.dqn.remember(state, action, reward, next_state, done)
                state = next_state
            if len(self.dqn.memory) > batch_size:
                self.dqn.replay(batch_size)