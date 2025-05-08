## cite from Visiability~
###IMPORTANT:   this code Do NOT use pytorch, using numpy
### ONLY for testing
import torch
import numpy as np
import cv2

from torch import tensor

global_map = np.zeros((1, 1))### just for avoid error(global_map = grid)
grid = np.zeros((1, 1))### just for avoid error(grid = np.zeros((1, 1)))

##test map

H,W = 100, 100
grid = np.ones((H,W),dtype=np.uint8)
cv2.rectangle(grid,(40,40),(75,75),0,-1)
global_map = grid

def display2map(map_origin, ratio, x_d):
    if len(x_d) == 2:
        x_d = np.array([[1, 0], [0, 1], [0, 0]]) @ x_d
    d2m = np.array([[0, 1 / ratio, 0],
                    [-1 / ratio, 0, 0],
                    [0, 0, 1]])
    return d2m @ (x_d - map_origin)

def map2display(map_origin, ratio, x_m):
    m2d = np.array([[0, -ratio, 0],
                    [ratio, 0, 0],
                    [0, 0, 1]])
    return m2d @ x_m + map_origin

import numpy as np

def SE2_kinematics(x, action, tau):
    wt_2 = action[1] * tau / 2
    t_v_sinc_term = tau * action[0] * np.sinc(wt_2 / np.pi)
    ret_x = np.empty(3)
    ret_x[0] = x[0] + t_v_sinc_term * np.cos(x[2] + wt_2)
    ret_x[1] = x[1] + t_v_sinc_term * np.sin(x[2] + wt_2)
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
def SDF_RT(robot_pose, fov, radius, RT_res, grid, inner_r=10):
    global global_map
    global_map = grid
    pts = raytracing(robot_pose, fov, radius, RT_res)
    x0, y0, theta = robot_pose
    x1_inner = x0 + inner_r * np.cos(theta - 0.5 * fov)
    y1_inner = y0 + inner_r * np.sin(theta - 0.5 * fov)
    x2_inner = x0 + inner_r * np.cos(theta + 0.5 * fov)
    y2_inner = y0 + inner_r * np.sin(theta + 0.5 * fov)
    pts = [[x1_inner, y1_inner]] + pts + [[x2_inner, y2_inner], [x1_inner, y1_inner]]
    return vertices_filter(np.array(pts))


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

def vertices_filter(polygon, angle_threshold=0.05):
    diff = polygon[1:] - polygon[:-1]
    diff_norm = np.sqrt(np.einsum('ij,ji->i', diff, diff.T))
    unit_vector = np.divide(diff, diff_norm[:, None], out=np.zeros_like(diff), where=diff_norm[:, None] != 0)
    angle_distance = np.round(np.einsum('ij,ji->i', unit_vector[:-1, :], unit_vector[1:, :].T), 5)
    angle_abs = np.abs(np.arccos(angle_distance))
    minimum_polygon = polygon[[True] + list(angle_abs > angle_threshold) + [True], :]
    return minimum_polygon

def polygon_SDF(polygon, point):
    N = len(polygon) - 1
    e = polygon[1:] - polygon[:-1]
    v = point - polygon[:-1]
    pq = v - e * np.clip((v[:, 0] * e[:, 0] + v[:, 1] * e[:, 1]) /
                         (e[:, 0] * e[:, 0] + e[:, 1] * e[:, 1]), 0, 1).reshape(N, -1)
    d = np.min(pq[:, 0] * pq[:, 0] + pq[:, 1] * pq[:, 1])
    wn = 0
    for i in range(N):
        val3 = np.cross(e[i], v[i])
        i2 = int(np.mod(i + 1, N))
        cond1 = 0 <= v[i, 1]
        cond2 = 0 > v[i2, 1]
        wn += 1 if cond1 and cond2 and val3 > 0 else 0
        wn -= 1 if ~cond1 and ~cond2 and val3 < 0 else 0
    sign = 1 if wn == 0 else -1
    return np.sqrt(d) * sign



### triabld_SDF may be used in the future for optimal
### normalize_angle() may have to change in the future as robot need to turn for many times
