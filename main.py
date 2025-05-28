import numpy as np
import cv2
from envs.simple_env  import *
#from simple_env_test import *
from utilities.utils import *
#from pursuer.planner.planner import *



def main():
    ## use json for map
    json_path = "0a1a5807d65749c1194ce1840354be39.json"
    grid = load_houseexpo_json_as_grid(json_path, canvas_size=(256, 256))

    psi = np.pi / 2
    radius = 50
    tau = 0.5
    env = SimpleEnv(grid, psi, radius, tau)

    ref_action = np.array([1.0, 0.0]) 
    for _ in range(100):
        tgt, rbt = env.update(ref_action)
        env.cv_render()

    env.close()

if __name__ == '__main__':
    main()

