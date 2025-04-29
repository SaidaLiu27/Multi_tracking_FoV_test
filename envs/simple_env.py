### cite from RL_active~

import torch
import cv2
import numpy as np
import matplotlib
matplotlib.Use('TKAgg')
import matplotlib.pyplot as plt

from typing import Tuple
from torch import tensor

## from utilities.utils import ちょっとわからんから一旦

class SimpleEnv:
    
    def __init__(self,num_landmarks, horizon, width, height, tau, A, B, V, W, 
                 landmark_motion_scale, psi, radius):
        

