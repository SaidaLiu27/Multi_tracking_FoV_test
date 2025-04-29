## cite from Visiability~

import torch
import numpy as np

from torch import tensor

global_map = np.zeros((1, 1))### just for avoid error(global_map = grid)
grid = np.zeros((1, 1))### just for avoid error(grid = np.zeros((1, 1)))


def SE2_kinematics(x: tensor, action: tensor, tau: float) -> tensor:
    wt_2 = action[1] * tau / 2
    t_v_sinc_term = tau * action[0] * torch.sinc(wt_2 / torch.pi)
    ret_x = torch.empty(3)
    ret_x[0] = x[0] + t_v_sinc_term * torch.cos(x[2] + wt_2)
    ret_x[1] = x[1] + t_v_sinc_term * torch.sin(x[2] + wt_2)
    ret_x[2] = x[2] + 2 * wt_2
    return ret_x

### occlusion of FoV , include steps until the visibility conflict the occlusion
def DDA(x0,y0,x1,y1):
    dx = x1 - x0
    dy = y1 - y0

    steps = int(max(abs(dx), abs(dy)))

    ## calculate the increment in x and y direction

    xinc = dx/ steps
    yinc = dy/ steps

    ## start from the first point
    x = float(x0)
    y = float(y0)

    for i in range(steps): ## what is global_map??
        if 0 < int(x)< len(global_map) and 0 < int(y) < len(global_map[0]):
            if global_map[int(x)][int(y)] == 0:
                break ## if the point is not occupied
        x += xinc
        y += yinc
    return int(x)+1, int(y)+1

##scope of FoV
def SDF_RT(robot_pose,fov,inner_r=10):
    global global_map
    global_map = grid
    pts = raytracing()
    x0,y0,theta = robot_pose
    x1_inner = x0 + inner_r * np.cos(theta - 0.5 * fov)
    y1_inner = y0 + inner_r * np.sin(theta - 0.5 * fov)
    x2_inner = x0 + inner_r * np.cos(theta + 0.5 * fov)
    y2_inner = y0 + inner_r * np.sin(theta + 0.5 * fov)
    ## pts is the list of points in the FoV(where the robot can see)
    pts = [[x1_inner, y1_inner]] + pts + [[x2_inner, y2_inner], [x1_inner, y1_inner]]
    return vertices_filter(np.array(pts))## vertices_filter is not defined in the code yet


def raytracing(robot_pose, fov, radius, RT_res):
    x0, y0, theta = robot_pose
    x1 = x0 + radius * np.cos(theta - 0.5 * fov)
    y1 = y0 + radius * np.sin(theta - 0.5 * fov)
    x2 = x0 + radius * np.cos(theta + 0.5 * fov)
    y2 = y0 + radius * np.sin(theta + 0.5 * fov)
    # y_mid = [y0 + radius * np.sin(theta - 0.5 * fov + i*fov / RT_res) for i in range(RT_res+1)]
    # x_mid = [x0 + radius * np.cos(theta - 0.5 * fov + i*fov / RT_res) for i in range(RT_res+1)]
    y_mid = np.linspace(y1, y2, RT_res)
    x_mid = np.linspace(x1, x2, RT_res)
  

    pts = []
    for i in range(len(x_mid)):
        xx, yy = DDA(int(x0), int(y0), int(x_mid[i]), int(y_mid[i]))
        if not pts or (yy != pts[-1][1] or xx != pts[-1][0]): pts.append([xx, yy])
    return pts

### 一応扇形に近似したものを実装するけど，多分使わへん気がする．
def raytracing_arc(robot_pose, fov, radius, RT_res):
    x0, y0, theta = robot_pose

    angles = np.linspace(theta - 0.5 * fov, theta + 0.5 * fov, RT_res)

    pts = []
    for angle in angles:
        x1 = x0 + radius * np.cos(angle)
        y1 = y0 + radius * np.sin(angle)

        xx, yy = DDA(int(x0), int(y0), int(x1), int(y1))

        if not pts or (yy != pts[-1][1] or xx != pts[-1][0]):## ここがあんまし分からんけどええか．
            pts.append([xx, yy])

    return pts