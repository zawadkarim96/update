import pandas as pd
import numpy as np
from ai_module import AIModule, TradingEnv
from indicators import get_all_indicators
from backtester import backtest
from config import OPTIMIZATION_MIN_ROWS, OPTIMIZATION_INTERVAL, RL_ENV_PARAMS
import datetime

class LearningEngine:
    def __init__(self):
        self.ai = AIModule()

    def optimize_strategies(self, historical_df: pd.DataFrame):
        """
        Optimize strategy parameters and update the AI models based on historical data.

        This function first checks that there is sufficient data for optimization. It then
        enriches the historical data with technical indicators, ensuring that price
        columns are retained. A subset of indicator columns is selected for training
        a genetic algorithm to find the best parameters, and a reinforcement learning
        (RL) model is optionally trained if the necessary libraries are available. A
        backtest is performed on a holdout set, and an LLM prompt is generated to
        aid further optimization.
        """
        if len(historical_df) < OPTIMIZATION_MIN_ROWS:
            print("Insufficient data for optimization.")
            return
        # Compute indicators and retain price columns so models can access OHLCV data
        historical_df = get_all_indicators(historical_df, include_price=True)
        # Split into train/test for walk‑forward optimization
        train_df = historical_df.iloc[:-1000]
        test_df = historical_df.iloc[-1000:]
        # Select a few informative features; ensure they exist in the DataFrame
        feature_cols = [col for col in ['TA_RSI_14', 'TA_MACD', 'TA_BBANDS_MIDDLE'] if col in train_df.columns]
        if not feature_cols:
            print("No valid feature columns for optimization. Skipping.")
            return
        X = train_df[feature_cols]
        # Define classification target based on next period return
        y = pd.cut(train_df['Close'].pct_change().shift(-1), bins=[-np.inf, -0.01, 0.01, np.inf], labels=[-1, 0, 1])
        # Optimize using a genetic algorithm; handle missing library gracefully
        try:
            best_params = self.ai.genetic_optimize(X, y)
        except Exception as e:
            print(f"Genetic optimization failed: {e}")
            best_params = None
        # Initialize and train RL model if available
        try:
            env = TradingEnv(train_df)
            self.ai.init_rl_model(env)
            if self.ai.rl_model:
                self.ai.rl_model.learn(total_timesteps=10000)
        except Exception as e:
            print(f"RL training skipped: {e}")
        # Perform a backtest on the test set using a baseline strategy
        backtest_result = backtest(test_df, 'moving_average_crossover')
        # Use the LLM (if available) to reflect on backtest results
        prompt = (
            f"Analyze backtest results and suggest optimizations. "
            f"Sharpe: {backtest_result['sharpe']}, "
            f"Win Rate: {backtest_result['win_rate']}, "
            f"Profit Factor: {backtest_result['profit_factor']}, "
            f"Max Drawdown: {backtest_result['max_drawdown']:.4f}, "
            f"Equity curve tail: {backtest_result['equity_curve'].tail(5).tolist()}"
        )
        reflection = self.ai.llm_decision_support(prompt)
        print(f"Optimization complete: {best_params}, Reflection: {reflection}")

    def self_reflect(self, historical_df: pd.DataFrame, threshold_win_rate=0.6):
        """
        Evaluate recent performance and trigger optimization if necessary.

        This method backtests a baseline strategy on the full historical data, computes
        the win rate and total profit/loss, and if performance falls below a
        threshold, triggers strategy optimization, evolution via a genetic
        algorithm, and initializes a DQN for further training.
        """
        # Backtest a baseline strategy (e.g. moving_average_crossover) on the entire data
        result = backtest(historical_df, 'moving_average_crossover')
        # Compute win/loss statistics from returns
        returns = result['equity_curve'].pct_change().fillna(0)
        positive_trades = (returns > 0).sum()
        total_trades = len(returns[returns != 0])
        win_rate = positive_trades / total_trades if total_trades > 0 else 0
        total_pl = result['equity_curve'].iloc[-1] - result['equity_curve'].iloc[0]
        print(f"Self-Reflection: Win Rate: {win_rate:.2f}, Total P/L: {total_pl:.2f}")
        if win_rate < threshold_win_rate or total_pl < 0:
            print("Performance below threshold. Triggering optimization and evolution.")
            self.optimize_strategies(historical_df)
            # Attempt to evolve strategies using genetic algorithm
            try:
                self.ai.evolve_strategies(historical_df)
            except Exception as e:
                print(f"Strategy evolution skipped: {e}")
            # Initialize and train a DQN if possible
            try:
                self.ai.init_dqn(RL_ENV_PARAMS['state_size'], RL_ENV_PARAMS['action_size'])
                if self.ai.dqn:
                    self.ai.train_dqn(TradingEnv(historical_df))
            except Exception as e:
                print(f"DQN training skipped: {e}")

    def nightly_retrain(self, historical_df):
        now = datetime.datetime.now()
        if now.hour == 3 and now.minute == 24 and OPTIMIZATION_INTERVAL == "daily":
            self.optimize_strategies(historical_df)
            self.self_reflect(historical_df)