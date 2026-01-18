import pytest
import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
import gymnasium_sudoku


def test_env_creation():
    try: # Test v0 with biased mode (should work)
        env = gym.make("sudoku-v0", mode="biased")
        print("created successfully")
        env.close()
    except Exception as e:
        print(f"creation failed:{e}")
    
    try: # Test v0 with easy mode (should fail)
        env = gym.make("sudoku-v0", mode="easy")
        print("should have failed but didn't")
        env.close()
    except ValueError as e:
        print(f"rejected mode='easy':{e}")
    
    try: # Test v1 with easy mode (should work)
        env = gym.make("sudoku-v1", mode="easy")
        print("created successfully")
        env.close()
    except Exception as e:
        print(f"creation failed: {e}")
    
    try: # Test v1 with biased mode (should fail)
        env = gym.make("sudoku-v1", mode="biased")
        env.close()
    except ValueError as e:
        print(f"rejected mode='biased':{e}")


def test_env_checker():
    try: # test v0
        env = gym.make("sudoku-v0",mode="biased")
        check_env(env.unwrapped)
        print("sudoku-v0 passes environment checker")
        env.close()
    except Exception as e:
        print(f"sudoku-v0 failed environment checker:{e}")
    
    try: # test v1
        env = gym.make("sudoku-v1",mode="easy")
        check_env(env.unwrapped)
        print("sudoku-v1 passes environment checker")
        env.close()
    except Exception as e:
        print(f"sudoku-v1 failed environment checker:{e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])



