>[!warning]
>  Under active development...Expect frequent code changes....

```
pip install gymnasium_sudoku
```

```python
import gymnasium_sudoku
import gymnasium as gym

env = gym.make("sudoku-v0",render_mode="human",horizon=150,render_delay=0.01,eval_mode=True)
env.reset() 
steps = 100

for n in range(steps):
    env.step(env.action_space.sample())
    env.render() 
```

And for training mode: 

```python
env = gym.make("sudoku-v0",horizon=150,eval_mode=False) # no rendering during  training  
```

**Observation space :** The state returned after each `.reset()` or `.step()` is a raw sudoku board shape `[9,9]`.This observation can be converted into an image.

**Action space:** The action space is shaped `[x,y,z]`,representing : x = row position of the cell, y = column position of the cell and value that should go into that cell.When vectorizing, the current version of the environment do not handle action reshaping, so for n environments, the action's shape should be : `[[x0...xn],[y0...yn],[z0...zn]]`

By default, `eval_mode` is set to `False`, this is good for training since after each reset() call,the Sudoku board will be changed to add more diversity to the training data and try to prevent memorization, so that the policy learns a more general distribution...At least that is the intuition.

During testing, `eval_mode` should be set to `True` to test the generalization capabilities of a trained policy or to test the environment with a random policy.This is important; otherwise,when testing a trained policy,it will be tested on states seen during training which would measure memorization rather than generalization.This makes it an invalid test of the policy's true capabilities.
