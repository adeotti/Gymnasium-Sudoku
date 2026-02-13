```
pip install gymnasium_sudoku
```

### Observation space
The state returned after each `.reset()` or `.step()` is a raw sudoku board shape `[9,9]`.This observation can be converted into an image.

### Action space 
The action space is shaped `[x,y,z]`,representing : x = row position of the cell, y = column position of the cell and value that should go into that cell.When vectorizing, the current version of the environment do not handle action reshaping, so for n environments, the action's shape should be : `[[x0...xn],[y0...yn],[z0...zn]]`

### Horizon 
This parameter controls the number of steps after which `Truncated` is set to `True` and the environment is reset. Otherwise, early in training (when the policy is still mostly random and the exploration incentive is high), the policy may corrupt the board and either make it unsolvable or push it into a local minimum. The default value for this parameter is set to 400 for no specific reason and should probably be adjusted during initialization.

### Eval mode/Training mode
By default, eval_mode in the __init__ method is set to `False`. This is used for training, where the environment is reset with one of 50 different boards after each .reset() call. During testing, eval_mode should be set to `True` in order to evaluate the agent on boards that were never seen during the training phase.

### Reward 
**biased environment (v0) :** The reward structure in the biased version (v0) penalizes the policy when it attempts to modify an untouchable cell (a given). In this version, the solution is deliberately exposed to the policy: the policy receives a positive reward when the inferred value matches the solution at the selected cell.This design choice is intentional and meant for rapid experimentation and debugging. In addition, the policy receives extra rewards when a row, column, or region is completed.

**unbiased environement (v1) :** The reward structure in the unbiased version (v1) penalizes the policy only when it attempts to modify an untouchable cell. Otherwise, the default reward is zero.
The policy is rewarded when the inferred number is unique within the region of the selected cell, defined as the union of its row, column, and 3×3 block, excluding the selected cell itself ([row+col+region]- index of the infered cell). Additional rewards are given when the inferred number is unique within the row, column, or region individually.

This reward design may appear redundant, but it is not. Early in training, the policy behaves almost randomly and frequently fills cells with incorrect values. In such cases, the condition “unique in [row + column + region] excluding the selected cell” can often be satisfied even when the inferred value is not the true solution. This reflects the intrinsic difficulty and the beauty.
Local minima lurk at each step: many moves are locally consistent yet lead to dead ends later in the episode. A successful policy must therefore learn not only to make locally valid moves, but also to recover from earlier mistakes and escape these local minima.


### Sudoku-v0 (biased version)
```python 
import gymnasium as gym

env = gym.make("sudoku-v0",mode="biased"render_mode="human",horizon=600,eval_mode=True)
env.reset() 

for n in range(int(6e3)):
    env.step(env.action_space.sample())
    env.render() 
```
**Bias :**
Among the induced biases that immensely help guide that learning is the fact that the policy cannot modify a cell that was already correctly filled, on top of the existing untouchable cells present in the beginning.

**Measuring learning :** The current structure of the environment allows a completely random policy to solve it (this is true for easy boards in the current version of the environment), so a good way to measure learning might be to use the number of steps over N episodes under a random policy as a `baseline`. This implies that a policy able to consistently solve the test boards in fewer steps over the same N episodes used to run a random policy is, in theory, displaying some sort of learning.


### Sudoku-v1
```python 
import gymnasium as gym

env = gym.make("sudoku-v1",mode="easy",render_mode="human",horizon=600,eval_mode=True)
env.reset() 

for n in range(int(6e3)):
    env.step(env.action_space.sample())
    env.render() 
```

**Difficulty :** This version allows the policy to corrupt the board during both training and testing. A previously filled cell can be modified with either an incorrect or a correct value as many times as the horizon allows. This characteristic makes learning significantly harder, since the agent is no longer operating in a setting where states only improve. One intuition for designing policies capable of solving this version of the environment is to build a policy that can recover from errors. For example, a policy might benefit from being able to run internal simulations in a latent state or directly in the tangible state before committing to a move. This is just a design intuition; other design philosophies might be even more effective.

**Measuring Learning :** Measuring learning here is quite straightforward and maybe the accent should be put on solving ability instead of speed, of course speed is important but to start, being able to solve different never seen boards from this version of the environment will be a sign of learning.


### data sources:
* Biased version data [source](https://www.kaggle.com/datasets/rohanrao/sudoku)
* Easy version data [source](https://www.kaggle.com/datasets/bryanpark/sudoku)
