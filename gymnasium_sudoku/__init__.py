from gymnasium.envs.registration import register
from gymnasium_sudoku.environment import Gym_env,V0_MODES,V1_MODES

__all__ = ["Gym_env"]
__version__ = "0.2.3"


def _make_v0(**kwargs):
    if not kwargs.get("mode") in V0_MODES:
        raise ValueError(f"sudoku-v0 requires mode {[*V0_MODES]}")
    return Gym_env(**kwargs)

def _make_v1(**kwargs):
    if not kwargs.get("mode") in V1_MODES:
        raise ValueError(f"sudoku-v1 availables modes are : {[*V1_MODES]}")
    return Gym_env(**kwargs)


register(
    id="sudoku-v0",
    entry_point="gymnasium_sudoku:_make_v0",
    kwargs={"mode":"biased"}
)

register(
    id="sudoku-v1",
    entry_point="gymnasium_sudoku:_make_v1",
    kwargs={"mode":"easy"}
)




