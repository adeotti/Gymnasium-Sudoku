import csv,random,sys
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces

from PySide6.QtWidgets import QApplication
from gymnasium_sudoku.rendering import Gui
from copy import deepcopy
from pathlib import Path
from collections import deque

def _get_region(x,y,board,n = 3):
    # T = target cell
    # returns the region (row - T ∪ column - T ∪ 3X3 block T)
    board = board.copy()
    xlist = board[x]
    xlist = np.concatenate((xlist[:y],xlist[y+1:]))

    ylist = board[:,y]
    ylist = np.concatenate((ylist[:x],ylist[x+1:]))
    
    ix,iy = (x//n)* n , (y//n)* n
    block = board[ix:ix+n , iy:iy+n].flatten()
    local_row = x - ix
    local_col = y - iy
    action_index= local_row * n + local_col
    block = np.delete(block,action_index)
    return xlist,ylist,block 

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
                 horizon=400,
                 eval_mode:bool=False,
                 rendering_attention=False
        ):
        super().__init__()
  
        self.env_mode = mode
        self.render_mode = render_mode
        self.horizon = horizon
        self.eval_mode = eval_mode
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
        self.conflicts = (self.state==0).sum()

        # init gui
        self.app = None
        if self.render_mode=="human":
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication([])
            
            self.gui = Gui(deepcopy(self.state),self.env_mode,self.rendering_attention)

    def _get_constrains_memory(self,board):
        # returns all rows, columns and regions on the board
        rows = [n.tolist() for n in board]
        cols = [m.tolist() for m in board.T] 
        regions = board.reshape(3,3,3,3).transpose(0,2,1,3).reshape(9,9)
        regions = [a.tolist() for a in regions] 
        memory = np.concatenate([rows,cols,regions])
        return memory

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.state,self.solution = _gen_board(self.env_mode,self.eval_mode)
        self.env_steps = 0
        self.mask = (self.state==0)
        self.conflicts_memory = []
        
        if self.render_mode =="human":
            self.gui.reset(deepcopy(self.state))
        return np.array(self.state,dtype=np.int32),{}

    def _get_biased_mode_reward(self):
        pass

    def _get_easy_mode_reward(self):
        pass

    def _get_reward(self,env_mode,action,state): 
        x,y,value = action

        if self.env_mode=="biased":
            if not self.mask[x,y]: 
                reward = -0.1 
                true_action = False  
            else:
                if value == self.solution[x,y]:
                    state[x,y] = value
                    self.mask[x,y] = False
                    assert action[-1] in range(1,10)
                    true_action = True  
                    reward = 0.2
                    
                    if _is_row_complete(state,x):
                        reward+= 0.2*9
                    if _is_col_complete(state,y):
                        reward+= 0.2*9
                    if _is_region_complete(state,x,y):
                        reward+= 0.2*9
                else:
                    reward = -0.1
                    true_action = False

            return reward,true_action,state

        elif env_mode=="easy":
            reward = 0
            if not self.mask[x][y]:
                reward = -0.1
                true_action = False
                return reward,true_action,state
            
            state[x][y] = value
            true_action = True
            filter_zeros = lambda x : x[x!=0]
            xlist,ylist,block = _get_region(x,y,state)

            row = filter_zeros(xlist)
            col = filter_zeros(ylist)
            block = filter_zeros(block)
            
            if not value in np.concatenate((xlist,ylist,block)):
                reward = 0.2*3
                return reward,true_action,state
            
            reward = 0
            if len(row) == len(np.unique(row)):
                reward += 0.2
            
            if len(col) == len(np.unique(col)):
                reward += 0.2
            
            if len(block) == len(np.unique(block)):
                reward += 0.2

            constrains_memory = self._get_constrains_memory(state)
            return reward,True,state     

    def step(self,action):
        assert (action[0] and action[1]) in range(9)
        self.env_steps+=1
        self.action = action
     
        reward,true_action,obs = self._get_reward(self.env_mode,self.action,self.state)
        self.true_action = true_action
        self.state = obs

        truncated = (self.env_steps>=self.horizon)
        done = np.array_equal(self.state,self.solution)
        if done:
            reward+=0.2*81
        info = {}
        return np.array(self.state,dtype=np.int32),round(reward,1),done,truncated,info

    def render(self):
        if self.render_mode == "human":
            self.gui.show()
            self.gui.updated(self.action,self.true_action)
            self.app.processEvents() 
        else:
            raise ValueError("Set render mode to human before calling .render()")

