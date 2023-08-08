#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
from pathlib import Path
from datetime import datetime
import json
from argparse import ArgumentParser

import numpy as np
import cv2

try: 
    sys.path.append("/home/jonathan/CARLA-10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg")
    import carla
except:
    quit()

import open3d as o3d

# Global settings
TOWN: int = 1
START_FRAME: int = 50

SPAWN_POINTS = [
    carla.Location(25.0, 326.5, 1.0),
    carla.Location(-1.5, 301.0, 1.0),
    carla.Location(30.0, -2.0, 1.0),
    carla.Location(394.0, 300.0, 1.0),
    carla.Location(397.0, 300.0, 1.0),
    carla.Location(396.5, 38.0, 1.0),
    carla.Location(346.0, 2.0, 1.0),
    carla.Location(30.0, 0.0, 1.0),
    carla.Location(350.0, 330.0, 1.0),
]


def visualize_image(image):
    data = np.array(image.raw_data)  # shape is (image.height * image.width * 4,)
    data_reshaped = np.reshape(data, (image.height, image.width, 4))
    rgb_3channels = data_reshaped[:, :, :3]  # first 3 channels

    cv2.imshow("image", rgb_3channels)
    cv2.waitKey(10)


def main(route_id: int, save_dir: str):
    actor_list = []
    save_dir = Path(save_dir)

    # Create output directory with current timestamp
    out_dir = save_dir / f"sequence_{route_id}_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    out_dir.mkdir(parents=True)

    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(30.0)
        world = client.get_world()
        world = client.load_world("/Game/Carla/Maps/Town01")

        # Switch to synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter("vehicle.bmw.*")[0]

        # Get spawn point
        carla_map = world.get_map()
        spawn_point = carla_map.get_waypoint(
            SPAWN_POINTS[route_id],
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        transform = spawn_point.transform
        transform.location.z += 0.2

        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print("created %s" % vehicle.type_id)

        # Create sensors
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("fov", str(100.0))
        camera_bp.set_attribute("image_size_x", str(512))
        camera_bp.set_attribute("image_size_y", str(256))

        camera_left_transform = carla.Transform(carla.Location(x=1.5, y=-0.2, z=1.5))
        camera_left = world.spawn_actor(camera_bp, camera_left_transform, attach_to=vehicle)
        actor_list.append(camera_left)
        print("created %s" % camera_left.type_id)

        camera_center_transform = carla.Transform(carla.Location(x=0.0, y=-0.2, z=1.5))
        camera_center = world.spawn_actor(camera_bp, camera_center_transform, attach_to=vehicle)
        actor_list.append(camera_center)
        print("created %s" % camera_center.type_id)

        camera_right_transform = carla.Transform(carla.Location(x=1.5, y=-0.2, z=1.5))
        camera_right = world.spawn_actor(camera_bp, camera_right_transform, attach_to=vehicle)
        actor_list.append(camera_right)
        print("created %s" % camera_right.type_id)

        lidar_rotation_frequency = 1.0 / settings.fixed_delta_seconds
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(64))
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / settings.fixed_delta_seconds))
        lidar_bp.set_attribute("points_per_second", str(lidar_rotation_frequency * 100_000))
        lidar_bp.set_attribute("range", str(20.0))
        lidar_bp.set_attribute("lower_fov", str(-45))
        lidar_transform = carla.Transform(carla.Location(z=3.0))
        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        actor_list.append(lidar)
        print(f"created {lidar.type_id}")

        frame = 0

        rgb_left_list = list()
        rgb_right_list = list()
        rgb_center_list = list()
        lidar_list = list()
        labels = list()

        camera_left.listen(
            lambda image: rgb_left_list.append(image) if frame > START_FRAME else None
        )
        camera_center.listen(
            lambda image: rgb_center_list.append(image) if frame > START_FRAME else None
        )
        camera_right.listen(
            lambda image: rgb_right_list.append(image) if frame > START_FRAME else None
        )
        lidar.listen(lambda pc: lidar_list.append(pc) if frame > START_FRAME else None)

        rgb_left_save_path = out_dir / "rgb_left"
        rgb_left_save_path.mkdir()
        rgb_center_save_path = out_dir / "rgb_center"
        rgb_center_save_path.mkdir()
        rgb_right_save_path = out_dir / "rgb_right"
        rgb_right_save_path.mkdir()
        lidar_save_path = out_dir / "lidar"
        lidar_save_path.mkdir()

        for frame in range(1_100):
            # Do tick
            world.tick()
            vehicle.set_autopilot(True)

            if frame > START_FRAME:
                visualize_image(rgb_left_list[-1])
                lidar_out = lidar_list[-1]

                # Save LiDAR point cloud
                pc = np.copy(np.frombuffer(lidar_out.raw_data, dtype=np.float32))
                pc = pc.reshape((len(pc) // 4, 4))
                pc = pc[:, :3]
                pcd = o3d.t.geometry.PointCloud()
                pcd.point["positions"] = pc
                o3d.t.io.write_point_cloud(str(lidar_save_path / f"frame_{frame:04d}.ply"), pcd)

                rgb_out = rgb_left_list[-1]
                rgb_out.save_to_disk(str(rgb_left_save_path / f"frame_{frame:04d}.png"))

                rgb_out = rgb_center_list[-1]
                rgb_out.save_to_disk(str(rgb_center_save_path / f"frame_{frame:04d}.png"))

                rgb_out = rgb_right_list[-1]
                rgb_out.save_to_disk(str(rgb_right_save_path / f"frame_{frame:04d}.png"))

            # Ensure that traffic lights are green
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                print(traffic_light)
                if (
                    traffic_light.get_state() == carla.TrafficLightState.Red
                    or traffic_light.get_state() == carla.TrafficLightState.Yellow
                ):
                    traffic_light.set_state(carla.TrafficLightState.Green)

            print("frame %s" % frame)

    finally:
        print("destroying actors")
        camera_left.destroy()
        camera_right.destroy()
        lidar.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print("done.")


if __name__ == "__main__":
    ap = ArgumentParser("Data Generation Script")
    ap.add_argument("route_id", type=int, help="Route ID")
    ap.add_argument("save_dir", type=str, help="Save directory")
    args = ap.parse_args()

    main(route_id=args.route_id, save_dir=args.save_dir)
