### cite from RL_active~

import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from utilities.utils import *
from typing import Tuple
from torch import tensor

## from utilities.utils import ちょっとわからんから一旦 <- 解決
## for testing SimpleEnv is way easy than class Environment
class SimpleEnv:
    def __init__(self, grid, psi, radius, tau=0.1):
        self._global_map = grid
        self._psi = psi
        self._radius = radius
        self._tau = tau
        self._rbt = np.array([30, 30, 0.0])
        self._tgt = np.array([80, 80, 0.0])
        self.map_ratio = 1
        #self.map_origin = np.array([[grid.shape[0]], [grid.shape[1] // 2], [np.pi / 2]])
        self.map_origin = np.array([[grid.shape[1]],   
                            [0],    
                            [0]])
        self._video_writer = None
        self.reset()

    def reset(self):
        self._rbt = np.array([30, 30, 0.0])
        self._tgt = np.array([80, 80, 0.0])
        height, width = self._global_map.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._video_writer = cv2.VideoWriter('./output.mp4', fourcc, 10.0, (width, height))
    
    
    def update(self, action):
        self._rbt = SE2_kinematics(self._rbt, action, self._tau)
        return self._tgt, self._rbt

    def get_visible_region(self):
        rbt_d = map2display(self.map_origin, self.map_ratio, self._rbt.reshape((3, 1))).squeeze()
        rt_visible = SDF_RT(rbt_d, self._psi, self._radius, 50, self._global_map)
        return rt_visible
    
    def sdf(self):
        visible_region = self.get_visible_region()
        return polygon_SDF(visible_region,self._tgt[0:2])
    
    def cv_render(self, save_path=None):
        canvas = cv2.cvtColor((self._global_map*225).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # --- FoV ポリゴン（display座標のまま）---
        poly_disp = self.get_visible_region().astype(np.int32)
        cv2.polylines(canvas, [poly_disp.reshape(-1,1,2)], True, (0,255,255), 1)

        # --- ターゲット・ロボットを display 座標に変換して描く ---
        tgt_d = map2display(self.map_origin, self.map_ratio, self._tgt.reshape(3,1)).squeeze()
        rbt_d = map2display(self.map_origin, self.map_ratio, self._rbt.reshape(3,1)).squeeze()
        cv2.circle(canvas, (int(tgt_d[0]), int(tgt_d[1])), 2, (0,0,255), -1)
        cv2.circle(canvas, (int(rbt_d[0]), int(rbt_d[1])), 2, (255,0,0), -1)
        

        cv2.imshow("Simulation", canvas)
        cv2.waitKey(1)

        if self._video_writer is not None:
            frame_bgr = cv2.resize(canvas,(canvas.shape[1],canvas.shape[0]))
            self._video_writer.write(frame_bgr)

    def close(self):
        if self._video_writer:
            self._video_writer.release()
        cv2.destroyAllWindows()
    
    

    # def render(self):
    #     vis_region = self.get_visible_region()
    #     canvas = cv2.cvtColor(self._global_map * 255, cv2.COLOR_GRAY2BGR)
    #     vis_region = vis_region.astype(np.int32)
    #     cv2.polylines(canvas, [vis_region.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
    #     cv2.circle(canvas, tuple(self._tgt[:2].astype(int)), 3, (0, 0, 255), -1)
    #     cv2.circle(canvas, tuple(self._rbt[:2].astype(int)), 3, (255, 0, 0), -1)
    #     cv2.imshow("FoV", canvas)
    #     cv2.waitKey(1)
        
        
        

