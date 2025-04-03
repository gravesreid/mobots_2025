import numpy as np
from typing import Tuple, List

from image_transform_reverse import real_coords_to_pixel_coords, get_warped_image
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt

def calculate_cos_sim(pose, cam_image, map, out_scale, image_size, calibration_pix, calibration_loc, sum_cam):
    x = pose[0]
    y = pose[1]
    theta = pose[2]
    warped_image = get_warped_image(x, y, theta, map, out_scale=out_scale, image_size=image_size, calibration_pix=calibration_pix, calibration_loc=calibration_loc)
    denom = np.linalg.norm(warped_image) * np.sqrt(sum_cam)
    if denom == 0:
        return 0
    cos_sim = np.sum(warped_image*cam_image) / denom
    # print(f"cos_sim: {cos_sim}")
    return cos_sim

class MobotLocator:
    MAP_PATH = "map_processing/final_path.png"
    PIXEL_SIZE = 0.03 # 3 cm
    ORIGIN_PIXEL = (2215.0, 178.0)
    CALIBRATION_PIXELS = [(1142, 629), (245, 239), (1025, 145), (1878, 276)]
    CALIBRATION_LOCS = [(0.34, 0), (0.68, 0.34), (1.02, 0), (0.68, -0.34)]
    CALIB_IMAGE_SIZE = (1920, 1080)

    def __init__(self, max_detlas: np.ndarray, step_size: np.ndarray, dist_penalty: float = 0.2, debug_print: bool = False):
        self.map = cv2.imread(self.MAP_PATH)
        self.map = cv2.cvtColor(self.map, cv2.COLOR_BGR2GRAY)
        self.max_detlas = max_detlas
        self.step_size = step_size
        self.dist_penalty = dist_penalty
        self.debug_print = debug_print

    def locate_image(self, cam_image: np.ndarray, x: float, y: float, theta: float) -> Tuple[float, float]:
        """
        Locate the image in the map.
        
        Args:
            cam_image: The image from the camera
            x: The x position of the camera in the map
            y: The y position of the camera in the map
            theta: The angle of the camera in the map (degrees)
            
        Returns:
            The x, y position of the image in the map
        """
        init_pose = np.array([x, y, theta])
        # make sure the cam_image has the same aspect ratio as the calibration image
        assert cam_image.shape[0] / cam_image.shape[1] == self.CALIB_IMAGE_SIZE[1] / self.CALIB_IMAGE_SIZE[0], f"Aspect ratio of the camera image ({cam_image.shape[0] / cam_image.shape[1]}) does not match the calibration image ({self.CALIB_IMAGE_SIZE[0] / self.CALIB_IMAGE_SIZE[1]})"

        out_scale =  cam_image.shape[0]/self.CALIB_IMAGE_SIZE[1]

        cam_image = np.array(cam_image, dtype=np.bool)
        sum_cam = np.sum(cam_image)

        # first, make srue the cam_image is in the right format
        # Get the warped image from the camera view
        
        max_delta = self.max_detlas
        step = self.step_size
        dir = np.array([1, 1, 1])
        new_sims = np.array([0.0, 0.0, 0.0])
        pose = np.array([x, y, theta])
        # for loop
        last_improve = np.array([1, 1, 1])

        last_cos_sim = calculate_cos_sim(pose, cam_image, self.map, out_scale, self.CALIB_IMAGE_SIZE, self.CALIBRATION_PIXELS, self.CALIBRATION_LOCS, sum_cam)
        
        for _ in range(20):
            if last_cos_sim == 0:
                # pick a random pose in the search space:
                for i in range(3):
                    pose[i] = np.random.uniform(-max_delta[i], max_delta[i]) + init_pose[i]
                last_cos_sim = calculate_cos_sim(pose, cam_image, self.map, out_scale, self.CALIB_IMAGE_SIZE, self.CALIBRATION_PIXELS, self.CALIBRATION_LOCS, sum_cam)
            else:
                if self.debug_print:
                    print('initial pose found')
                break
        else:
            if self.debug_print:
                print("No pose found")
            return 0, 0, 0
        
        t_max = 25
        for t in range(t_max):
            test_poses = []
            dist_penalties = np.zeros(3)
            for i in range(3):
                test_poses.append(pose.copy())
                test_poses[i][i] += dir[i] * step[i]
                test_poses[i] = np.clip(test_poses[i], init_pose - max_delta, init_pose + max_delta)
                percent_dist = np.linalg.norm((test_poses[i] - init_pose)/max_delta)/np.sqrt(3)
                # print('percent_dist: ', percent_dist)
                dist_penalties[i] = 1 - self.dist_penalty * percent_dist # range from 1 to 1- dist_penalty

            new_sims[0] = calculate_cos_sim(test_poses[0], cam_image, self.map, out_scale, self.CALIB_IMAGE_SIZE, self.CALIBRATION_PIXELS, self.CALIBRATION_LOCS, sum_cam)
            new_sims[1] = calculate_cos_sim(test_poses[1], cam_image, self.map, out_scale, self.CALIB_IMAGE_SIZE, self.CALIBRATION_PIXELS, self.CALIBRATION_LOCS, sum_cam)
            new_sims[2] = calculate_cos_sim(test_poses[2], cam_image, self.map, out_scale, self.CALIB_IMAGE_SIZE, self.CALIBRATION_PIXELS, self.CALIBRATION_LOCS, sum_cam)

            # apply dist penalties
            new_sims = new_sims*dist_penalties

            improve = new_sims > last_cos_sim
            if self.debug_print:
                print('dist_penalties: ', dist_penalties)
                print('last_cos_sim: ', last_cos_sim)
                print("new_sims: ", new_sims)
                print("improve: ", improve)

            if np.sum(improve) + np.sum(last_improve) == 0:
                break

            pose = pose + dir * (improve * step)

            pose = np.clip(pose, init_pose - max_delta, init_pose + max_delta)

            dir = dir * (-1 + 2*improve) # if improve is true, dir *= 1, if false, dir *= -1
           
            last_cos_sim = np.max(new_sims)
            last_improve = improve.copy()
            
        # # multiprocessing
        # with Pool(4) as p:
        #     cos_sims = p.starmap(calculate_cos_sim, [(x + dx, 
        #                                               y + dy, 
        #                                               theta + dtheta, 
        #                                               cam_image, 
        #                                               self.map, 
        #                                               out_scale, 
        #                                               self.CALIB_IMAGE_SIZE, 
        #                                               self.CALIBRATION_PIXELS, 
        #                                               self.CALIBRATION_LOCS, sum_cam) for dx, dy, dtheta in combos])
            
        print(f"cos_sims: {last_cos_sim}")

        return pose - init_pose
    
    def pose_to_pixel(self, pose: np.ndarray, scale = np.array([1, 0.65])) -> Tuple[int, int]:
        """
        Convert the pose to pixel coordinates.
        
        Args:
            pose: The x, y position of the camera in the map. If the pose is a 1D array, it is assumed to be a single pose. If it is a 2D array, it is assumed to be multiple poses.
            
        Returns:
            The pixel coordinates of the pose
        """
        pose = pose/scale
        rank = len(pose.shape)
        if rank == 1:
            out = np.zeros(2)
            out[0] = self.ORIGIN_PIXEL[0] - pose[0] / self.PIXEL_SIZE
            out[1] = self.ORIGIN_PIXEL[1] + pose[1] / self.PIXEL_SIZE
        else:
            out = np.zeros((pose.shape[0], 2))
            out[:, 0] = self.ORIGIN_PIXEL[0] - pose[:, 0] / self.PIXEL_SIZE
            out[:, 1] = self.ORIGIN_PIXEL[1] + pose[:, 1] / self.PIXEL_SIZE

        return out
        
    
    def render_sim_image(self, pose: np.ndarray, cam_image: np.ndarray) -> np.ndarray:
        """
        Render the simulated image from the camera view.
        
        Args:
            pose: The x, y, theta position of the camera in the map
            cam_image: The image from the camera
            
        Returns:
            The simulated image from the camera view
        """
        x, y, theta = pose

        out_scale =  cam_image.shape[0]/self.CALIB_IMAGE_SIZE[1]
        warped_image = get_warped_image(x, y, theta, self.map, out_scale, image_size=self.CALIB_IMAGE_SIZE, calibration_pix=self.CALIBRATION_PIXELS, calibration_loc=self.CALIBRATION_LOCS)
        
        # plot the warped image in red and the cam_image in blue overlayed
        combo_image = np.zeros((cam_image.shape[0], cam_image.shape[1], 3), dtype=np.uint8)
        combo_image[:, :, 0] = cam_image
        combo_image[:, :, 2] = warped_image

        return combo_image

from image_thresh import thresh_image
if __name__ == "__main__":
    # Load the map image
    # Create the MobotLocator object
    locator = MobotLocator(max_detlas=np.array([0.1, 0.1, 10]), step_size=np.array([0.01, 0.01, 1.]))

    # Load the camera image
    cam_image = cv2.imread("data/old_images/image_20250401_180328_386.jpg")

    # image 640 x 480. Make it 270 x 480
    cam_image = cv2.resize(cam_image, (480, 270))

    cam_mask = thresh_image(cam_image)

    x0 = 2
    y0 = -0.2
    theta0 = -10

    # Locate the image in the map
    dx, dy, dtheta = locator.locate_image(cam_mask, x0, y0, theta0)

    print(f"dx: {dx}, dy: {dy}, dtheta: {dtheta}")

    render_image = locator.render_sim_image(np.array([x0 + dx, y0 + dy, theta0 + dtheta]), cam_mask)

    plt.figure()
    plt.imshow(render_image)
    plt.show()