import numpy as np

class MobotLocator:
    MAP_PATH = "/home/aigeorge/projects/mobots_2025/map_processing/final_path.png"
    PIXEL_SIZE = 0.03 # 3 cm
    ORIGIN_PIXEL = (2418.8, 175.3)
    
    def __init__(self, map: np.ndarray, pixel_size: float, origin_pixel: Tuple[int, int]):
        self.map = map
        self.pixel_size = pixel_size
        self.origin_pixel = origin_pixel