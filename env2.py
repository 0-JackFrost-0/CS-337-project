import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env


# Constants
WINDOW_SIZE = 0
INITIAL_BALANCE = 100000
TRADING_FEE = 0.01
MAX_MOVE = 200
INIT_SHARES = 0


class TradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(TradingEnv, self).__init__()
        # Drop first 2 columns

        self.data =  data.iloc[:, 1:].drop(columns=["open","tic","day","high","low"])
        # print(self.data.head())
        self.dates = data.iloc[:, 0]
        self.initial_balance = INITIAL_BALANCE
        self.balance = self.initial_balance
        self.position = INIT_SHARES  # Number of stocks held
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.balance
        self.portfolio_value_history = [self.portfolio_value]
        self.action = 0.1
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float64)

        dict = {}
        for column in self.data.columns:
            dict[column] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
        dict["balance"] = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32)
        dict["position"] = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.int32)
        dict["portfolio_value"] = spaces.Box(
            low=0, high=np.inf, shape=(1,), dtype=np.float32)
        dict["prev_action"] = spaces.Box(
            low=-1, high=1, shape=(1,), dtype=np.float64)
        
        self.observation_space = spaces.Dict(dict)

    def reset(self, seed=None):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.balance
        next_observation = self._next_observation()
        info = self._next_info()
        # next_observation = np.array(list(next_observation.values()))
        return next_observation, info

    def _buy(self, action):
        current_price = self.data.iloc[self.current_step]["close"]
        amount_to_buy = min(np.floor(MAX_MOVE * action),
                            self.balance*(1-TRADING_FEE)//current_price)
        trading_fee = amount_to_buy * current_price * TRADING_FEE
        self.position += amount_to_buy
        self.balance -= amount_to_buy * current_price + trading_fee
        # print("Buy: ", amount_to_buy, " at ", current_price)

    def _sell(self, action):
        action = -action
        current_price = self.data.iloc[self.current_step]["close"]
        amount_to_sell = np.floor(self.position * action)
        trading_fee = amount_to_sell * current_price * TRADING_FEE
        self.balance += amount_to_sell * current_price - trading_fee
        self.position -= amount_to_sell
        # print("Sell: ", amount_to_sell, " at ", current_price)

    def calculate_portfolio_value(self, current_price):
        return self.balance + self.position * current_price

    def step(self, action:np.float32):
        self.action = action
        current_price = self.data.iloc[self.current_step]["close"]

        if action == 0:
            pass
        elif action > 0:
            self._buy(action)
        else:
            self._sell(action)

        self.portfolio_value = self.calculate_portfolio_value(current_price)
        reward = self._get_reward()
        self.portfolio_value_history.append(self.portfolio_value)
        obs = self._next_observation()
        self.current_step += 1
        self.done = self.current_step >= len(self.data) - 1
        info = self._next_info()

        return obs, reward, self.done, False, info

    def _next_observation(self):
        obs = self.data.iloc[self.current_step]
        obs = obs.to_dict()
        for key in obs.keys():
            obs[key] = np.array([obs[key]])
        assert self.balance >= 0
        obs["balance"] = np.array([self.balance], dtype=np.float32).reshape(1,)
        obs["position"] = np.array([self.position], dtype=np.int32).reshape(1,)
        obs["portfolio_value"] = np.array([self.portfolio_value], dtype=np.float32).reshape(1,)
        obs['prev_action'] = np.array([self.action], dtype=np.float32).reshape(1,)
        return obs
    
    def _next_info(self):
        info = {}
        info["portfolio_value"] = self.portfolio_value
        info["balance"] = self.balance
        info["position"] = self.position
        info["current_step"] = self.current_step
        info["action"] = self.action
        info["date"] = self.dates.iloc[self.current_step]
        info["close"] = self.data.iloc[self.current_step]["close"]
        return info

    def _get_reward(self):
        portfolio_value_before = self.portfolio_value_history[-1]
        reward = (self.portfolio_value - portfolio_value_before)
        # Cast to simple int
        reward = int(reward)
        return reward

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    df = pd.read_csv("data/RELIANCE.NS.csv")
    env = TradingEnv(df)
    check_env(env)
    obs, info = env.reset()
    print("###################")
    print("Date: ", info["date"])
    print("Action: ", info["action"])
    print("Balance: ", info["balance"])
    print("Position: ", info["position"])
    print("Portfolio Value: ", info["portfolio_value"])
    print("###################")
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, __, info = env.step(action)
        print("###################")
        print("Date: ", info["date"])
        print("Action: ", info["action"])
        print("Balance: ", info["balance"])
        print("Position: ", info["position"])
        print("Portfolio Value: ", info["portfolio_value"])
        print("###################")
        