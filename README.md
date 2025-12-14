>[!warning]
> Actively under developpement

usage :

```
pip install gymnasium_sudoku
```

```python
import gymnasium_sudoku
import gymnasium as gym

env = gym.make("sudoku-v0",render_mode="human")
env.reset()
steps = 100

for n in range(steps):
    env.step(env.action_space.sample())
    env.render()
```
