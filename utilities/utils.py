## cite from Visiability~
###IMPORTANT:   this code Do NOT use pytorch, using numpy
### ONLY for testing
import torch
import numpy as np
import cv2
import json

from torch import tensor

global_map = np.zeros((1, 1))### just for avoid error(global_map = grid)
grid = np.zeros((1, 1))### just for avoid error(grid = np.zeros((1, 1)))

##test map

H,W = 100, 100
grid = np.ones((H,W),dtype=np.uint8)
cv2.rectangle(grid,(40,40),(75,75),0,-1)
global_map = grid

def display2map(map_origin, ratio, x_d):
    
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
            if global_map[int(y)][int(x)] == 0:
                break ## if the point is not occupied
        x += xinc
        y += yinc
    return int(x)+1, int(y)+1

# def DDA(x0, y0, x1, y1):
#     dx, dy = x1 - x0, y1 - y0
#     steps = int(max(abs(dx), abs(dy)))
#     if steps == 0:                             # 同一点クリック対策
#         return x0, y0

#     xinc, yinc = dx/steps, dy/steps
#     x, y = float(x0), float(y0)

#     for _ in range(steps):
#         row, col = int(round(x)), int(round(y))     # ← **行(row)=y, 列(col)=x**
#         if 0 <= row < len(global_map) and 0 <= col < len(global_map[0]):
#             if grid[row, col] == 0:                 # 黒セル命中
#                 return col, row                     # 直前のセルを返す
#         x += xinc
#         y += yinc
#     return int(round(x)), int(round(y))

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


### FoV penerate only one sell
##  easy to fix but, the model codes is return int(x+1), int(y+1)
## dont know should be change or not


## function for use HouseExpo dataset(.json)
## so import .json -> grid map 
## cite from test.py

### ----- Problem: in test.py, the y axis is normal but 
### in envs/simple_env.py, the y axis is flipped
### ---Solution: #polygon[:, 1] = canvas_size[1] - polygon[:, 1]
def load_houseexpo_json_as_grid(json_path, canvas_size=(256, 256)):
    with open(json_path, 'r') as f:
        data = json.load(f)

    grid = np.zeros(canvas_size, dtype=np.uint8)

    bbox_min = data['bbox']['min']
    bbox_max = data['bbox']['max']
    width = bbox_max[0] - bbox_min[0]
    height = bbox_max[1] - bbox_min[1]
    scale_x = canvas_size[0] / width
    scale_y = canvas_size[1] / height
    scale = min(scale_x, scale_y)
    ## なんかめっちゃおかしい気がするけどとりあえずうまく動くからええわ
    if "verts" in data:
        polygon = np.array([
            [int((v[0] - bbox_min[0]) * scale), int((v[1] - bbox_min[1]) * scale)]
            for v in data["verts"]
        ], dtype=np.int32)
        #polygon[:, 1] = canvas_size[1] - polygon[:, 1]
        cv2.fillPoly(grid, [polygon], color=255)


    ## 0 is obstacle, 255 is free space 
    ## I think 255 is white and 0 is black but in this code,
    ## fill polygon with 255(black) and 0(white)
    ## 意味わからんけど，とりあえずうまく動くからええわ
    return (grid == 255).astype(np.uint8)