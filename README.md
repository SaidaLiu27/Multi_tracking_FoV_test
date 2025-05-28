# Multi-agent Tracking with Occlusioned Fov

## Code overview
- `main.py` — Entry point to run simulation
- `envs/simple_env.py` — Simulation environment (decide robot position, render image, .mp4)

- `utilities/utils.py` - Simulation utilities (decide Fov, load map (load_houseexpo_json_as_grid) )

- `test.py` - test code for making grid map from json file

## code explanation
 - ` utilities/utils.py`
    - `DDA(x0, y0, x1, y1)`
        - determine visibility until lay hit an obstacle.
    - `SDF_RT(robot_pose, psi, radius, ...)`
        - constracts a visibility polygon from now robot pose and map
    - ` map2display(), display()`
        - convert grid map coordinate and display coordinate

 - ` envs/simple_env.py`

 - ` test.py`
    - import json file, and make a grid map which is all black(0)
    - in json file, "verts" defines verticle of wall 
    - sampling verticle with pixel(int())
    - fill with white (cv2.fillpoly())
    