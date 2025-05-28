import json
import numpy as np
import cv2

def load_houseexpo_json(json_path, canvas_size=(256, 256)):
    with open(json_path, 'r') as f:
        data = json.load(f)

    grid = np.zeros(canvas_size, dtype=np.uint8)
    
    ## code added for describe all
    bbox_min = data['bbox']['min']
    bbox_max = data['bbox']['max']
    width = bbox_max[0] - bbox_min[0]
    height = bbox_max[1] - bbox_min[1]
    scale_x = canvas_size[0] / width
    scale_y = canvas_size[1] / height
    scale = min(scale_x, scale_y)

    if "verts" in data:
        polygon = np.array([
            [int((v[0] - bbox_min[0]) * scale), int((v[1] - bbox_min[1]) * scale)]
            for v in data["verts"]
        ], dtype=np.int32)

        ##idk why y axis is flipped code under isnot needed
        #polygon[:, 1] = canvas_size[1] - polygon[:, 1]

        ### obstacles are black and spaces are white
        cv2.fillPoly(grid, [polygon], color=255)

    return grid

## sample usage
json_path = "0a1a5807d65749c1194ce1840354be39.json"  
grid_img = load_houseexpo_json(json_path)

# display the image(map)
#cv2.imshow("HouseExpo Map", grid_img)
#cv2.waitKey(0)
cv2.imwrite("test_output_2.png", grid_img)
#cv2.destroyAllWindows()