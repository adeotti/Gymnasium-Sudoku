>[!warning]
>  Under active development...Expect frequent code changes....

```
pip install gymnasium_sudoku
```

```python
import gymnasium_sudoku
import gymnasium as gym

env = gym.make("sudoku-v0",render_mode="human",horizon=150,eval_mode=True)
env.reset(delay=0.1) # delay param control the update rate of the gui
steps = 100

for n in range(steps):
    env.step(env.action_space.sample())
    env.render() 
```

And for training : 

```python
env = gym.make("sudoku-v0",horizon=150,eval_mode=False)
# It is better not to call .render() during training 
```


By default, `eval_mode` is set to `False`, this is good for training since after each reset() call,the Sudoku board will be changed to add more diversity to the training data and try to prevent memorization, so that the policy learns a more general distribution...At least that is the intuition.

During testing, `eval_mode` should be set to `True` to test the generalization capabilities of a trained policy or to test the environment with a random policy.This is important; otherwise,when testing a trained policy,it will be tested on states seen during training which would measure memorization rather than generalization.This makes it an invalid test of the policy's true capabilities.
