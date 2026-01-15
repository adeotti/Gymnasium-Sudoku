* **Observation space :** The state returned after each `.reset()` or `.step()` is a raw sudoku board shape `[9,9]`.This observation can be converted into an image.

* **Action space:** The action space is shaped `[x,y,z]`,representing : x = row position of the cell, y = column position of the cell and value that should go into that cell.When vectorizing, the current version of the environment do not handle action reshaping, so for n environments, the action's shape should be : `[[x0...xn],[y0...yn],[z0...zn]]`


### Sudoku-v0 (biased version)
The latest version of this environment is 0.2.3

* **Installation :**
```python
import gymnasium_sudoku<=0.2.3
import gymnasium as gym

env = gym.make("sudoku-v0",render_mode="human",horizon=150,render_delay=1.0,eval_mode=True)
env.reset() 

for n in range(int(6e3)):
    env.step(env.action_space.sample())
    env.render() 
```
**Eval mode/Training mode :** By default, `eval_mode` in the init method is set to `False`, this is for training where the envrionment will be reseted with 100 differents boards after each .reset() calls.During testing, `eval_mode` should be set to `True` to test on board never seen during the training phase.The booards using during training and test are in separate csv files : sudoku_100.csv and sudoku_50.scv (100 boards in the training file and 50 in the test file)

**Bias :**
Among the induced biases that immensely help guide that learning is the fact that the policy cannot modify a cell that was already correctly filled, on top of the existing untouchable cells present in the beginning.

**Measuring learning for this version of the environment:*** The current structure of the environment allows a completely random policy to solve it (this is true for easy boards in the current version of the environment), so a good way to measure learning might be to use the number of steps over N episodes under a random policy as a `baseline`. This implies that a policy able to consistently solve the test boards in fewer steps over the same N episodes used to run a random policy is, in theory, displaying some sort of learning.


### Sudoku-v1
