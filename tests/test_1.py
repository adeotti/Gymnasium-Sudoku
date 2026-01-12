import gymnasium_sudoku
import gymnasium as gym
import time,sys
import numpy as np
from gymnasium_sudoku.puzzle import tb_1,ts_1
from tqdm import tqdm

env = gym.make(
        "sudoku-v0",
        render_mode="human",
        horizon=150,
        render_delay=0.01,
        eval_mode=False
    )

env.reset()
steps = int(6e3*5) 

for n in tqdm(range(steps),total=steps):
    obs,reward,done,trunc,info = env.step(env.action_space.sample())
    env.render()
    if done:
        print(f"\n{obs}")
        assert np.all(obs!=0)

        if env.unwrapped.eval_mode:
            assert np.array_equal(obs,ts_1)
        elif not env.unwrapped.eval_mode:
            assert np.array_equal(obs,env.unwrapped.solution)
        
        time.sleep(5) # delay the board reset
        env.reset()
