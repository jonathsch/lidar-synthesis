echo "Running Odometry..."
kiss_icp_pipeline $1/lidar/
mv ./results/latest/lidar_poses.npy $1/lidar_poses.npy

echo "Alignment and Meshing..."
python scripts/alignment.py $1/lidar $1/lidar_poses.npy
python scripts/meshing.py $1/pcd_accumulated.ply

echo "Augmenting..."
python scripts/augment.py $1