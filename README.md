>[!warning]
>  Under active development...Expect frequent code changes....

```
pip install gymnasium_sudoku>=0.1.10
```

```python
import gymnasium_sudoku
import gymnasium as gym

env = gym.make("sudoku-v0",render_mode="human",horizon=300)
env.reset()
steps = 100

for n in range(steps):
    env.step(env.action_space.sample())
    env.render() 
```

And for training : 

```python
env = gym.make("sudoku-v0",horizon=300)
# It is better not to call .render() during training 
```
