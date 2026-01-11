import time,sys,os,csv,random,torch
import numpy as np

from PySide6 import QtCore,QtGui
from PySide6.QtWidgets import QApplication,QWidget,QGridLayout,QLineEdit,QHBoxLayout
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon 

import gymnasium as gym
import gymnasium.spaces as spaces
from copy import deepcopy


def sudoku_board():
    with open("./sudoku_100.csv") as file:
        reader = csv.reader(file)
        randomint = random.randint(1,99)
        for n,row in enumerate(reader):
            if n == randomint:
                chosen_line = row
        board,solution = chosen_line
        board,solution = list(
                map(lambda x:np.fromiter(x,dtype=np.int32).reshape(9,9),(board,solution))
        )
    return board,solution
            

class Gui(QWidget):
    def __init__(self,board,rendering_attention=False):
        super().__init__()
        self.setWindowTitle("Sudoku")
        self.setMaximumSize(40,40)
        self.setWindowIcon(QIcon("game.png"))
        self.game = board
        self.size = 9
        self.rendering_attention = rendering_attention
    
        self.main_layout = QHBoxLayout()

        # Sudoku grid
        self.grid = QGridLayout()
        self.sudoku_widget = QWidget()
        self.sudoku_widget.setLayout(self.grid)
        self.main_layout.addWidget(self.sudoku_widget)
        self.grid.setVerticalSpacing(0)
        self.grid.setHorizontalSpacing(0)
        self.grid.setContentsMargins(0,0,0,0)


        self.cells = [[QLineEdit(self) for _ in range(self.size)] for _ in range (self.size)] 
        for line in self.game :
            for x in range(self.size):
                for y in range(self.size):
                    self.cells[x][y].setFixedSize(40,40)
                    self.cells[x][y].setReadOnly(True)
                    number = str(board[x][y])
                    self.cells[x][y].setText(number)
                    self.bl = (3 if (y%3 == 0 and y!= 0) else 0.5) # what is bl,bt ? 
                    self.bt = (3 if (x%3 == 0 and x!= 0) else 0.5)
                    self.color = ("transparent" if int(self.cells[x][y].text()) == 0 else "white")
                    self.cellStyle = [
                        "background-color:grey;"
                        f"border-left:{self.bl}px solid black;"
                        f"border-top: {self.bt}px solid black;"
                        "border-right: 1px solid black;"
                        "border-bottom: 1px solid black;"
                        f"color: {self.color};"
                        "font-weight: None;"
                        "font-size: 20px"
                    ]
                    self.cells[x][y].setStyleSheet("".join(self.cellStyle))
                    self.cells[x][y].setAlignment(QtCore.Qt.AlignCenter)
                    self.grid.addWidget(self.cells[x][y],x,y)
        
        if self.rendering_attention:
            # Attention grid
            self.attn_grid = QGridLayout()
            self.attn_widget = QWidget()
            self.attn_widget.setLayout(self.attn_grid)
            self.main_layout.addWidget(self.attn_widget)
            self.attn_grid.setVerticalSpacing(0)
            self.attn_grid.setHorizontalSpacing(0)
            self.attn_grid.setContentsMargins(0,0,0,0)

            self.attn_cells = [[QLineEdit(self) for _ in range(self.size)] for _ in range(self.size)]
            for x in range(self.size):
                for y in range(self.size):
                    cell = self.attn_cells[x][y]
                    cell.setFixedSize(40,40)
                    cell.setAlignment(QtCore.Qt.AlignCenter)
                    cell.setStyleSheet(
                        "background-color: black;"
                        "border:none;"
                    )
                    self.attn_grid.addWidget(cell, x, y)

        self.setLayout(self.main_layout)
 
    def updated(self,action:[int,int,int],true_value:bool=False,attention_weights=None) -> list[list[int]]: 
        if action is not None: 
            assert len(action) == 3
            row,column,value = action
            styleList = self.cells[row][column].styleSheet().split(";")
            if len(styleList) != 8 : # small bug fix here, more documentation maybe...
                del styleList[-1]
            styleDict = {k.strip() : v.strip() for k,v in (element.split(":") for element in styleList)}
            cellColor = styleDict["color"]

            if cellColor != "white" and cellColor != "black":
                assert value in range(1,10)
                self.cells[row][column].setText(str(value))   # Update cell with value
                self.game[row][column] = value                # Update grid with value
                color = ("transparent" if not true_value else "black")
                ubl = (3 if (column % 3 == 0 and column!= 0) else 0.5)
                ubt = (3 if (row % 3 == 0 and row!= 0) else 0.5)
                updatedStyle = [
                    "background-color:dark grey;"
                    f"border-left:{ubl}px solid black;"
                    f"border-top: {ubt}px solid black;"
                    "border-right: 1px solid black;"
                    "border-bottom: 1px solid black;"
                    f"color: {color};"
                    "font-weight: None;"
                    "font-size: 20px"
                ]
                self.cells[row][column].setStyleSheet("".join(updatedStyle)) # Update the cell color flash

                def reset_style():
                    background = "orange" if color == "black" else "grey"
                    normalStyle = [
                        f"background-color:{background};",
                        f"border-left:{ubl}px solid black;",
                        f"border-top: {ubt}px solid black;",
                        "border-right: 1px solid black;",
                        "border-bottom: 1px solid black;",
                        f"color: {color};",
                        "font-weight: None;",
                        "font-size: 20px;"
                    ]
                    self.cells[row][column].setStyleSheet("".join(normalStyle)) 

                QTimer.singleShot(20, reset_style)  # Delay in milliseconds
                
                styleList = self.cells[row][column].styleSheet().split(";")
                styleDict = {k.strip() : v.strip() for k,v in (element.split(":") for element in styleList)}
                cellColor = styleDict["color"] 

                if self.rendering_attention and attention_weights is not None:
                    self.render_attention(attention_weights)
            
        return self.game

    def reset(self,board):
        self.game = board
        for line in self.game :
            for x in range(self.size):
                for y in range(self.size):
                    self.cells[x][y].setFixedSize(40,40)
                    self.cells[x][y].setReadOnly(True)
                    number = str(board[x][y])
                    self.cells[x][y].setText(number)
                    self.bl = (3 if (y%3 == 0 and y!= 0) else 0.5) 
                    self.bt = (3 if (x%3 == 0 and x!= 0) else 0.5)
                    self.color = ("transparent" if int(self.cells[x][y].text()) == 0 else "white")
                    self.cellStyle = [
                        "background-color:grey;"
                        f"border-left:{self.bl}px solid black;"
                        f"border-top: {self.bt}px solid black;"
                        "border-right: 1px solid black;"
                        "border-bottom: 1px solid black;"
                        f"color: {self.color};"
                        "font-weight: None;"
                        "font-size: 20px"
                    ]
                    self.cells[x][y].setStyleSheet("".join(self.cellStyle))

    def render_attention(self,attn):
        for i in range(self.size):
            for j in range(self.size):
                v = attn[i, j]
                intensity = int(255 * v)
                self.attn_cells[i][j].setStyleSheet(
                    f"""
                    background-color: rgb({intensity}, {intensity}, 255);
                    """
                )
      

def region_fn(index:list,board,n = 3): # returns the region (row ∪ column ∪ 3X3 block) of a cells
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
    

app = QApplication.instance()
if app is None:
    app = QApplication([])


class Gym_env(gym.Env): 
    metadata = {"render_modes": ["human"],"render_fps":60,"rendering_attention":False}   
    def __init__(self,render_mode=None,horizon=100,rendering_attention=False):
        super().__init__()
        self.rendering_attention = rendering_attention
        self.horizon = horizon
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
        
        self.state,self.solution = deepcopy(sudoku_board())
        self.mask = (self.state==0)
        self.gui = Gui(deepcopy(self.state),self.rendering_attention)
        
        self.region = region_fn
        self.render_mode = render_mode
        self.conflicts = (self.state == 0).sum()
                
    def reset(self,seed=None, options=None) -> np.array :
        super().reset(seed=seed)
        self.state,self.solution = deepcopy(sudoku_board())
        self.env_steps = 0
        self.mask = (self.state == 0)
        
        if self.render_mode =="human":
            self.gui.reset(deepcopy(self.state))
        return np.array(self.state,dtype=np.int32),{}

    def step(self,action):
        assert (action[0] and action[1]) in range(9),f"x and y not in range [0,1, 2, 3, 4, 5, 6, 7, 8]"
        assert action[-1] in range(1,10),f"cell value not in range [1, 2, 3, 4, 5, 6, 7, 8, 9]"
        self.env_steps+=1
        self.action = action
        x,y,value = self.action 

        if not self.mask[x,y]: # if target cell is not modifiable
            reward = -0.1 
            self.true_action = False  
        else:
            if value == self.solution[x,y]:
                self.state[x,y] = value
                self.mask[x,y] = False
                self.true_action = True  
                reward = 0.3
                
                if _is_row_complete(self.state,x):
                    reward+= 0.3*9
                if _is_col_complete(self.state,y):
                    reward+= 0.3*9
                if _is_region_complete(self.state,x,y):
                    reward+= 0.3*9

            else:
                reward = -0.1
                self.true_action = False    

        truncated = (self.env_steps>=self.horizon)
        done = np.array_equal(self.state,self.solution)
        if done:
            reward+=0.3*81
            
        info = {}
        return np.array(self.state,dtype=np.int32),round(reward,1),done,truncated,info

    def render(self,delay = 0.1,attention_weights=None):
        if self.render_mode == "human":
             
            if attention_weights is not None and self.rendering_attention:
                self.state = self.gui.updated(self.action,self.true_action,attention_weights)
            else:
                self.gui.updated(self.action,self.true_action)
                
            self.gui.show()
            app.processEvents()
            time.sleep(delay)
        else :
            sys.exit("render_mode attribute should be set to \"human\"")





