from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import open3d as o3d


def main(lidar_path: str, pose_file: str, stride: int):
    lidar_path = Path(lidar_path)
    poses = np.load(pose_file)  # (N, 4, 4)
    lidar_paths = sorted([p for p in lidar_path.iterdir() if p.suffix == ".ply"])

    poses = poses[::stride]
    lidar_paths = lidar_paths[::stride]

    pc_acc = None

    for idx in range(len(poses)):
        pcd = o3d.t.io.read_point_cloud(str(lidar_paths[idx])).cpu()
        lidar = pcd.transform(poses[idx])
        lidar.point.positions = lidar.point.positions.to(o3d.core.float32)

        if pc_acc is None:
            pc_acc = lidar.clone()
        else:
            pc_acc = pc_acc.append(lidar.clone())

    out_path = lidar_path.parent / "pcd_accumulated.ply"
    o3d.t.io.write_point_cloud(str(out_path), pc_acc)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("lidar_path", type=str, help="Path to data directory")
    ap.add_argument("pose_file", type=str, help="Path to pose file")
    ap.add_argument("--stride", type=int, default=10, help="Stride for alignment")
    args = ap.parse_args()
    main(args.lidar_path, args.pose_file, args.stride)
