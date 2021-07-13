# 3D-Animal-Pose
3D pose prediction with multiple cameras and bundle adjustment

To create the necessary conda environment, run
`conda env create -f requirements.yml`

The above will create a conda environment called `3d-pose`, which you should `conda activate` before proceeding.

# NOTE: This is a work in progress
We appreciate any kind of feedback. Open an issue or send a message, and we'll try our best to implement that feedback

# Instructions
The script `reconstruct_points.py` is where the reconstruction happens. It takes as input a path to a config file, e.g. `./configs/ibl.yaml`. To perform reconstruction run the code as follows:

``python reconstruct_points.py --config ./configs/ibl.yaml --output_dir ./outputs/ --downsample 10 --reproj_thresh 6``

The `output_dir` argument specifies where the outputs are saved. `downsample` specifices the rate to downsample the frames before saving to video. This does NOT downsample the 3D points, only the frames for visualization. `reproj_thresh` specifies a threshold for selecting frames with high reprojection error. 

For a new dataset, make a new config file that follows the pattern of the ones in `./configs`. This requires specifying paths to the videos in each view, CSV files in each view, camera parameters (intrinsics and extrinsics), and bodypart information (bodypart names, colors, etc.).
