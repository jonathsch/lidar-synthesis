from collections import deque
from pathlib import Path
from PIL import Image

import carla
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation
import torch
from torchvision.transforms import ToTensor
import lightning.pytorch as pl

from lidar_synthesis.leaderboard.leaderboard.autoagents import autonomous_agent

from lidar_synthesis.utils.pc_utils import random_sampling
from lidar_synthesis.models.nvidia import LitImageToSteering


def get_entry_point():
    return "CameraAgent"


model_path = Path("/home/jonathan/dev/lidar-augmentation/logs/train/runs/2023-04-28_16-42-22/checkpoints/last.ckpt").resolve()
model = None


class CameraAgent(autonomous_agent.AutonomousAgent):
    """
    Dummy autonomous agent to control the ego vehicle
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.to_tensor = ToTensor()

        self.step = -1
        self.control = carla.VehicleControl()
        self.control.steer = 0.0
        self.control.throttle = 0.0
        self.control.brake = 1.0

        self.action_repeat = 2

        # LiDAR transformation
        self.lidar_transform = Rotation.from_euler("xyz", [0, 0, -90], degrees=True)

        ###############################
        # PID Controller
        ###############################
        turn_KP = 1.25
        turn_KI = 0.5
        turn_KD = 0.25
        turn_n = 3  # buffer size

        speed_KP = 5.0
        speed_KI = 0.5
        speed_KD = 1.0
        speed_n = 20  # buffer size

        self.default_speed = 3.0  # Speed used when creeping
        self.max_throttle = 0.3  # upper limit on throttle signal value in dataset
        self.brake_speed = 0.4  # desired speed below which brake is triggered
        self.brake_ratio = (
            1.1  # ratio of speed to desired speed at which brake is triggered
        )
        self.clip_delta = (
            0.25  # maximum change in speed input to logitudinal controller
        )
        self.clip_throttle = 0.4  # Maximum throttle allowed by the controller

        self.turn_controller = PIDController(
            K_P=turn_KP, K_I=turn_KI, K_D=turn_KD, n=turn_n
        )
        self.speed_controller = PIDController(
            K_P=speed_KP, K_I=speed_KI, K_D=speed_KD, n=speed_n
        )

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 1.5, 'y': 0.0, 'z': 1.50, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 512, 'height': 256, 'fov': 100, 'id': 'center'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}


        """
        ROTATION_FREQUENCY = 20.0
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "x": 1.5,
                "y": 0.0,
                "z": 1.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 512,
                "height": 256,
                "fov": 100,
                "id": "Center",
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 0.0,
                "y": 0.0,
                "z": 2.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": -90.0,
                "id": "lidar",
                "channels": 64,
                "rotation_frequency": ROTATION_FREQUENCY,
                "points_per_second": ROTATION_FREQUENCY * 100_000,
                "range": 20.0,
                "dropoff_general_rate": 0.0,
                "atmosphere_attenuation_rate": 0.0,
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": ROTATION_FREQUENCY,
                "id": "speed",
            },
        ]

        return sensors

    def _control_pid(self, waypoints, velocity, is_stuck):
        """Predicts vehicle control with a PID controller.

        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        """

        # waypoints[1, 2] *= 2.0
        waypoints[1, 0] = (1.0 / 3.0) * waypoints[1, 3]
        waypoints[1, 1] = (2.0 / 3.0) * waypoints[1, 3]
        waypoints[1, 2] = 0.8 * waypoints[1, 3]
        waypoints[1, :] -= 0.1

        # waypoints[1, :] *= -1
        waypoints = waypoints.T

        # when training we transform the waypoints to lidar coordinate, so we need to change is back when control
        # waypoints[:, 0] += self.config.lidar_pos[0]

        speed = velocity

        desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0

        if is_stuck:
            desired_speed = np.array(self.default_speed)  # default speed of 14.4 km/h

        brake = (desired_speed < self.brake_speed) or (
            (speed / desired_speed) > self.brake_ratio
        )

        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.clip_throttle)
        throttle = throttle if not brake else 0.0

        aim = (waypoints[1] + waypoints[0]) / 2.0
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01:
            angle = 0.0  # When we don't move we don't want the angle error to accumulate in the integral
        if brake:
            angle = 0.0

        steer = self.turn_controller.step(angle)

        steer = np.clip(steer, -0.5, 0.5)  # Valid steering values are in [-1,1]

        return steer, throttle, brake

    @torch.inference_mode()
    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """

        # print("=====================>")
        """
        for key, val in input_data.items():
            if hasattr(val[1], "shape"):
                shape = val[1].shape
                print("[{} -- {:06d}] with shape {}".format(key, val[0], shape))
            else:
                print("[{} -- {:06d}] ".format(key, val[0]))
        print("<=====================")
        """
        self.step += 1

        global model
        if model is None:
            model = LitImageToSteering.load_from_checkpoint(model_path)
            model.eval()
            model.to("cuda")

        # Log image
        image_buffer = np.copy(input_data["Center"][1][:, :, :3])
        # image = Image.fromarray(input_data["cam_center"][1][:, :, :3], "RGB")
        image = Image.frombuffer(
            "RGB", (512, 256), image_buffer, "raw", "RGB", 0, 1
        )
        image.save(f"frame_{self.step}.png")
        # image = np.array(image)
        # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("CARLA", image)

        if self.step % self.action_repeat == 1:
            return self.control

        # image.save(f"image_{self.step}.png")

        # Get LiDAR point cloud
        # lidar = np.copy(input_data["lidar"][1][:, :3])
        # lidar = self.lidar_transform.apply(lidar)

        # lidar = lidar[np.linalg.norm(lidar, axis=1) < 20.0]
        # lidar = lidar[lidar[:, 0] > 0.0]

        # Rotate into vehicle coordinate system

        # lidar = random_sampling(lidar, 16_384).astype(np.float32)

        # pcd_log = o3d.t.geometry.PointCloud()
        # pcd_log.point["positions"] = lidar
        # o3d.t.io.write_point_cloud(f"lidar_{timestamp}.ply", pcd_log)

        # lidar = torch.from_numpy(lidar.T).float().unsqueeze(0).to("cuda")

        # if lidar.max() > 25.0:
        #     print(
        #         "lidar max: ", lidar[:, 0].max(), lidar[:, 1].max(), lidar[:, 2].max()
        #     )
        image = self.to_tensor(image).unsqueeze(0).to("cuda")
        steering = model(image)[0].detach().cpu().numpy().item()
        # print(f"step: {self.step}, steering: {steering}")

        # waypoints = waypoints.reshape(2, 4)
        # # print(waypoints.T)

        # speed = input_data["speed"][1]["speed"]

        # steer, throttle, brake = self._control_pid(
        #     steering,
        #     0.4,
        #     True,
        # )

        # print(f"steer: {steer}, throttle: {throttle}, brake: {brake}")

        # Add noise to the control
        perturbation = 0.2
        p = np.random.uniform(0, 1)
        if p < 0.5:
            steering += np.random.uniform(-perturbation, perturbation)

        # RETURN CONTROL
        control = carla.VehicleControl()
        control.steer = steering
        control.throttle = 0.4
        control.brake = 0.0

        self.control = control

        return control


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)

    def step(self, error):
        self._window.append(error)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = self._window[-1] - self._window[-2]
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative
