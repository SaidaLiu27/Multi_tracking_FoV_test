import numpy as np
import cv2
from envs.simple_env  import *
from utilities.utils import *

#make test map 
def create_test_map():
    H, W = 100, 100
    grid = np.ones((H, W), dtype=np.uint8)
    cv2.rectangle(grid, (40, 40), (75, 75), 0, -1)
    return grid

def main():
    # パラメータ設定（YAMLは若rん）
    grid = np.ones((100, 100), dtype=np.uint8)
    cv2.rectangle(grid, (40, 40), (75, 75), 0, -1)
    psi = np.pi / 2
    radius = 60
    tau = 0.5

    env = SimpleEnv(grid, psi, radius, tau)
    ref_action = np.array([1.0, 0.0])  # 定速直進

    for step in range(100):
        tgt, rbt = env.update(ref_action)
        sdf_val = env.sdf()
        print(f"[Step {step}] Target SDF: {sdf_val:.3f} {'[VISIBLE]' if sdf_val < 0 else '[OCCLUDED]'}")
        
        save_file = f"test_output.png"
        env.cv_render(save_path=save_file)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()