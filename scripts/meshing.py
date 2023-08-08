from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pymeshlab


def main(pcd_path: str):
    pcd_path = Path(pcd_path).resolve()

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(pcd_path))
    ms.generate_simplified_point_cloud(samplenum=200_000)
    ms.set_current_mesh(1)
    ms.generate_surface_reconstruction_ball_pivoting()
    ms.save_current_mesh(str(pcd_path.parent / "lidar_mesh.ply"))


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("pcd_path", type=str, help="Path to accumulated point cloud")
    args = ap.parse_args()
    main(args.pcd_path)
