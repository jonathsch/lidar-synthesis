from pathlib import Path
import json
from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def raycast_mesh(
    mesh: o3d.t.geometry.TriangleMesh,
    channels: int = 32,
    n_measurements: int = 250,
    transform: np.array = None,
):
    num_rays = channels * n_measurements
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)

    rays = np.empty((n_measurements, channels, 6))
    horizontal_angles = np.linspace(-np.pi / 2, np.pi / 2, num=rays.shape[0])
    vertical_angles = -1 * powspace(np.pi / 64, np.pi / 3.0, power=2, num=rays.shape[1])

    ray_vertical_directions = np.array(
        [
            np.cos(vertical_angles),
            np.zeros_like(vertical_angles),
            np.sin(vertical_angles),
        ]
    ).T

    rays[:, :, :3] = np.array([0.0, 0.0, 1.5])
    orientation = transform[:3, :3]
    origin = orientation @ np.array([0.0, 0.0, 1.5]) + transform[:3, 3]

    for i in range(len(horizontal_angles)):
        for j in range(len(ray_vertical_directions)):
            rot = Rotation.from_euler("xyz", [0.0, 0.0, horizontal_angles[i]], degrees=False)
            direction = rot.apply(ray_vertical_directions[j])
            direction = orientation @ direction
            rays[i, j, :3] = origin
            rays[i, j, 3:] = direction

    rays = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    ans_dict = scene.cast_rays(rays)

    rays = rays.numpy()
    depth = ans_dict["t_hit"].numpy().reshape(-1)
    rays = rays.reshape((num_rays, 6))

    points = rays[:, :3] + (rays[:, 3:].T * depth).T
    points = points[np.isfinite(depth)]

    return points


def main():
    ap = ArgumentParser("Data augmentation utility")
    ap.add_argument("sequence", type=str)
    ap.add_argument("--buffer", type=int, default=50, dest="buffer")
    ap.add_argument("--samples", type=int, default=50, dest="samples")
    ap.add_argument("--deviation", type=float, default=2.0, dest="dev")
    ap.add_argument("-lo", "--labels-only", action="store_true", dest="labels_only")

    args = ap.parse_args()

    out_dir = Path(args.sequence, "raycast")
    out_dir.mkdir(exist_ok=True)

    kiss_poses = np.load(Path(args.sequence, "lidar_poses.npy"))

    mesh_file = Path(args.sequence, "lidar_mesh.ply")
    mesh = o3d.io.read_triangle_mesh(str(mesh_file))
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    augmented_labels = []

    for idx in tqdm(range(20, len(kiss_poses) - 20)):
        T_target = kiss_poses[idx]

        T_wp_target = kiss_poses[idx + 10]
        T_wp_gb_target = kiss_poses[idx + 15]

        trajectory_offsets = np.linspace(-args.dev, args.dev, args.samples, dtype=np.float32)

        for j, offset in enumerate(trajectory_offsets):
            rel_pose = T_target.copy()
            rel_pose[:3, 3] += rel_pose[:3, :3] @ np.array([0.0, offset, 0.0])
            inv_rel_pose = np.linalg.inv(rel_pose)

            T_curr = T_target
            tmp_pose_1 = T_curr @ np.linalg.inv(T_wp_target)
            tmp_pose_2 = T_curr @ np.linalg.inv(T_wp_gb_target)

            y_points = np.array(
                [
                    0,
                    (tmp_pose_1[:3, :3] @ tmp_pose_1[:3, 3])[1] - offset / 3,
                    (tmp_pose_2[:3, :3] @ tmp_pose_2[:3, 3])[1] - offset / 3,
                ]
            )

            x_points = np.array(
                [
                    0,
                    tmp_pose_1[0, 3],
                    tmp_pose_2[0, 3],
                ]
            )

            y_interp_fn = CubicSpline([0, 4, 5], y_points)
            x_interp_fn = interp1d([0, 4, 5], x_points, kind="linear")
            waypoints = np.array([x_interp_fn(np.arange(0, 5, 1)), y_interp_fn(np.arange(0, 5, 1))])

            if not args.labels_only:
                pc = raycast_mesh(mesh, channels=64, n_measurements=500, transform=rel_pose)
                pc = (inv_rel_pose[:3, :3] @ pc.T).T + inv_rel_pose[:3, 3]
                pc = pc[np.linalg.norm(pc, axis=1) <= 20.0]

            out_path = out_dir / f"raycast-{idx}-{j}.ply"

            augmented_labels.append(
                {
                    "waypoints": waypoints.tolist(),
                    "file": str(out_path.name),
                }
            )

            if not args.labels_only:
                pcd = o3d.t.geometry.PointCloud()
                pcd.point["positions"] = pc
                o3d.t.io.write_point_cloud(str(out_path), pcd)

    # Save generated labels as json file
    with open(out_dir / "augmented-labels.json", mode="w") as f:
        json.dump(augmented_labels, f)


if __name__ == "__main__":
    main()
