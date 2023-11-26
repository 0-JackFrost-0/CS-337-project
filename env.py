from copy import deepcopy

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

# Constants
WINDOW_SIZE = 0
INITIAL_BALANCE = 10000
TRADING_FEE = 0.01
MAX_MOVE = 200
INIT_SHARES = 0


class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(TradingEnv, self).__init__()
        self.data = data
        self.initial_balance = INITIAL_BALANCE
        self.balance = self.initial_balance
        self.position = INIT_SHARES  # Number of stocks held
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.balance
        self.portfolio_value_history = [self.portfolio_value]
        self.action = 0
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)

        dict = {}
        self.dummy_data = deepcopy(self.data)
        self.dummy_data.drop(columns=["date", 'day', 'tic'], inplace=True)
        for column in self.dummy_data.columns:
            dict[column] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        dict["balance"] = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32)
        dict["position"] = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.int32)
        dict["portfolio_value"] = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32)
        dict["prev_action"] = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Dict(dict)

    def reset(self, seed=None):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.balance
        next_observation = self._next_observation()
        info = self._next_info()
        return next_observation, info

    def buy(self, action):
        current_price = self.data.iloc[self.current_step]["close"]
        amount_to_buy = min(np.floor(MAX_MOVE * action),
                            self.balance//current_price)
        trading_fee = amount_to_buy * current_price * TRADING_FEE
        self.position += amount_to_buy
        self.balance -= amount_to_buy * current_price + trading_fee

    def sell(self, action):
        action = -action
        current_price = self.data.iloc[self.current_step]["close"]
        amount_to_sell = min(np.floor(MAX_MOVE * action), self.position)
        trading_fee = amount_to_sell * current_price * TRADING_FEE
        self.balance += amount_to_sell * current_price - trading_fee
        self.position -= amount_to_sell

    def calculate_portfolio_value(self, current_price):
        return self.balance + self.position * current_price

    def step(self, action):
        self.action = action
        current_price = self.data.iloc[self.current_step]["close"]

        if action == 0:
            pass
        elif action > 0:
            self.buy(action)
        else:
            self.sell(action)

        self.portfolio_value = self.calculate_portfolio_value(current_price)
        reward = self._get_reward()
        self.portfolio_value_history.append(self.portfolio_value)
        obs = self._next_observation()
        info = self._next_info()
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        return obs, reward, self.done, False, info

    def _next_observation(self):
        obs = self.dummy_data.iloc[self.current_step]
        obs = obs.to_dict()
        obs["balance"] = self.balance
        obs["position"] = self.position
        obs["portfolio_value"] = self.portfolio_value
        obs['prev_action'] = self.action
        l = []
        for k, v in obs.items():
            l.append(v)
        return l

    def _next_info(self):
        obs = self.dummy_data.iloc[self.current_step]
        obs = obs.to_dict()
        obs["balance"] = self.balance
        obs["position"] = self.position
        obs["portfolio_value"] = self.portfolio_value
        obs['prev_action'] = self.action
        obs['date'] = self.data.iloc[self.current_step]["date"]
        return obs

    def _get_reward(self):
        portfolio_value_before = self.portfolio_value_history[-1]
        reward = (self.portfolio_value - portfolio_value_before)
        return reward

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    df = pd.read_csv("data/RELIANCE.NS.csv")
    env = TradingEnv(df)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, __, ___ = env.step(action)
        print(obs)
        print(reward)
        print(done)
        # print(info)
        print()
