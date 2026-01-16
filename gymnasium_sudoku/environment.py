import csv,random
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces

from PySide6.QtWidgets import QApplication
from gymnasium_sudoku.rendering import Gui
from copy import deepcopy
from pathlib import Path

def _region_fn(index:list,board,n = 3): 
    # returns the region (row ∪ column ∪ 3X3 block) of a cells
    board = board.copy()
    x,y = index
    xlist = board[x]
    xlist = np.concatenate((xlist[:y],xlist[y+1:]))

    ylist = board[:,y]
    ylist = np.concatenate((ylist[:x],ylist[x+1:]))
    
    ix,iy = (x//n)* n , (y//n)* n
    block = board[ix:ix+n , iy:iy+n].flatten()
    local_row = x - ix
    local_col = y - iy
    action_index = local_row * n + local_col
    block = np.delete(block,action_index)
    return np.concatenate((xlist,ylist,block))

def _is_row_complete(board,x):
    xlist = board[x]
    return np.all(xlist!=0)

def _is_col_complete(board,y):
    ylist = board[:,y]
    return np.all(ylist!=0)

def _is_region_complete(board,x,y,n=3):
    ix,iy = (x//n)* n , (y//n)* n
    block = board[ix:ix+n , iy:iy+n].flatten()
    return np.all(block!=0)

def _sudoku_board(csv_path,line_pick): 
    with open(csv_path) as file:
        reader = csv.reader(file)
        for n,row in enumerate(reader):
            if n == line_pick:
                chosen_line = row
        board,solution = chosen_line
        board,solution = list(
                map(lambda x:np.fromiter(x,dtype=np.int32).reshape(9,9),(board,solution))
        )
    return board,solution

def _gen_board(env_mode,eval_mode):
    csv_path = Path(__file__).parent 
    if env_mode=="biased":
        csv_path_train = csv_path/"datasets/v0_biased/train_boards.csv"
        csv_path_test = csv_path/"datasets/v0_biased/test_boards.csv"
        line_pick = random.randint(0,49)
    
    elif env_mode=="easy":
        csv_path_train = csv_path/"datasets/v1_easy/train_boards.csv"
        csv_path_test = csv_path/"datasets/v1_easy/test_boards.csv"
        line_pick = random.randint(0,49)

    if eval_mode:        
        state,solution = deepcopy(_sudoku_board(csv_path_test,line_pick))
    else:
        state,solution = deepcopy(_sudoku_board(csv_path_train,line_pick))
    return state,solution


V0_MODES = ["biased"]
V1_MODES = ["easy"]

class Gym_env(gym.Env): 
    metadata = {"render_modes":["human"],"render_fps":60,"rendering_attention":False}   
    def __init__(self,
                 mode,
                 render_mode=None,
                 horizon=100,
                 eval_mode:bool=False,
                 render_delay:float=0.0,
                 rendering_attention=False
        ):
        super().__init__()
  
        self.env_mode = mode
        self.render_mode = render_mode
        self.horizon = horizon
        self.eval_mode = eval_mode
        self.render_delay = render_delay
        self.rendering_attention = rendering_attention
        self.env_steps = 0
        self.action = None
        self.true_action = False

        self.action_space = spaces.Tuple(
            (
            spaces.Discrete(9,None,0),
            spaces.Discrete(9,None,0),
            spaces.Discrete(9,None,1)
            )
        )
        self.observation_space = spaces.Box(0,9,(9,9),dtype=np.int32)

        self.state,self.solution = _gen_board(self.env_mode,self.eval_mode)
        self.mask = (self.state==0)
        self.region = _region_fn
        self.conflicts = (self.state==0).sum()

        # init gui
        self.app = None
        if self.render_mode=="human":
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])
            
            self.gui = Gui(deepcopy(self.state),self.rendering_attention)
 
    def reset(self,seed=None,options=None) -> np.array :
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.state,self.solution = _gen_board(self.env_mode,self.eval_mode)
        self.env_steps = 0
        self.mask = (self.state==0)
        
        if self.render_mode =="human":
            self.gui.reset(deepcopy(self.state))
        return np.array(self.state,dtype=np.int32),{}
    
    def _get_reward(self,env_mode,action): 
        x,y,value = action

        if self.env_mode=="biased":
            if not self.mask[x,y]: 
                reward = -0.1 
                true_action = False  
            else:
                if value == self.solution[x,y]:
                    self.state[x,y] = value
                    self.mask[x,y] = False
                    assert action[-1] in range(1,10)
                    true_action = True  
                    reward = 0.3
                    
                    if _is_row_complete(self.state,x):
                        reward+= 0.3*9
                    if _is_col_complete(self.state,y):
                        reward+= 0.3*9
                    if _is_region_complete(self.state,x,y):
                        reward+= 0.3*9
                else:
                    reward = -0.1
                    true_action = False
            return reward,true_action

        elif env_mode=="easy":
            pass 

    def step(self,action):
        assert (action[0] and action[1]) in range(9)
        self.env_steps+=1
        self.action = action
        
        reward,true_action = self._get_reward(self.env_mode,self.action)
        self.true_action = true_action

        truncated = (self.env_steps>=self.horizon)
        done = np.array_equal(self.state,self.solution)
        if done:
            reward+=0.3*81
            
        info = {}
        return np.array(self.state,dtype=np.int32),round(reward,1),done,truncated,info

    def render(self):
        self.gui.show()
        self.gui.updated(self.action,self.true_action,self.render_delay)
        self.app.processEvents() 


