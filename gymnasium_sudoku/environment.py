from gymnasium_sudoku.puzzle import easyBoard,solution
import time,sys
import numpy as np

from PySide6 import QtCore,QtGui
from PySide6.QtWidgets import QApplication,QWidget,QGridLayout,QLineEdit
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon 

import gymnasium as gym
import gymnasium.spaces as spaces
from copy import deepcopy

class Gui(QWidget):
    def __init__(self,board):
        super().__init__()
        self.setWindowTitle("Sudoku")
        self.setMaximumSize(20,20)
        self.setWindowIcon(QIcon("game.png"))
        self.game = board
        self.grid = QGridLayout(self)
        self.grid.setSpacing(0)
        self.size = 9
        self.cells = [[QLineEdit(self) for _ in range(self.size)] for _ in range (self.size)] 
        for line in self.game :
            for x in range(self.size):
                for y in range(self.size):
                    self.cells[x][y].setFixedSize(40,40)
                    self.cells[x][y].setReadOnly(True)
                    number = str(easyBoard[x][y])
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

    def updated(self,action:[int,int,int],true_value : bool = False) -> list[list[int]]: 
        if action is not None: 
            assert len(action) == 3
            row,column,value = action
            styleList = self.cells[row][column].styleSheet().split(";")
            if len(styleList) != 8 : # small bug fix here, more documentation maybe...
                del styleList[-1]
            styleDict = {k.strip() : v.strip() for k,v in (element.split(":") for element in styleList)}
            cellColor = styleDict["color"]

            if cellColor != "white" and cellColor != "black":
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
        return self.game


def region_fn(index:list,board,n = 3): # returns the region (row ∪ column ∪ 3X3 block) of a cells
    board = board.copy()
    x,y = index

    xlist = board[x]
    xlist = np.concatenate((xlist[:y],xlist[y+1:]))

    ylist = board[:,y]
    ylist = np.concatenate((ylist[:x],ylist[x+1:]))
    
    ix,iy = (x//n)* n , (y//n)* n
    block = board[ix:ix+n , iy:iy+n].flatten()
    # -
    local_row = x - ix
    local_col = y - iy
    action_index = local_row * n + local_col
    block = np.delete(block,action_index)
    return np.concatenate((xlist,ylist,block))


app = QApplication.instance()
if app is None:
    app = QApplication([])


class Gym_env(gym.Env): 
    metadata = {"render_modes": ["human"],"render_fps":60}   
    def __init__(self,render_mode = None,horizon = 300):
        super().__init__()
        self.horizon = horizon
        self.step = 0
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

        self.state = deepcopy(easyBoard)
        self.mask = (self.state==0)
        self.gui = Gui(self.state)
        self.region = region_fn
        self.render_mode = render_mode
        self.conflicts = (self.state == 0).sum()
                
    def reset(self,seed=None, options=None) -> np.array :
        super().reset(seed=seed) 
        self.state = deepcopy(easyBoard)
        self.step = 0
        self.mask = (self.state == 0)
        return np.array(self.state,dtype=np.int32),{}

    def step(self,action):    
        self.step+=1
        self.action = action
        x,y,value = self.action 

        if not self.mask[x,y]: # if target cell is not modifiable
            reward = -0.5
            self.true_action = False
            done = True if self.step == self.horizon else False
            return np.array(self.state,dtype=np.int32),reward,done,truncated,{}

        region = self.region((x,y),self.state)
        is_unique = not np.any(region==value).item()

        if is_unique: 
            self.state[x,y] = value
            self.mask[x,y] = False
            self.true_action = True  
            reward = 1
        else:
            reward = -0.5
            self.true_action = False
          
        info = {}
        done = True if self.step == self.horizon else False
        truncated = False
        return np.array(self.state,dtype=np.int32),reward,done,truncated,info

    def render(self):
        if self.render_mode == "human": 
            self.state = self.gui.updated(self.action,self.true_action)
            self.gui.show()
            app.processEvents()
            time.sleep(0.1)
        else :
            sys.exit("render_mode attribute should be set to \"human\"")


