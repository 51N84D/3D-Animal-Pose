# 3D-Animal-Pose
3D pose prediction with multiple cameras and bundle adjustment

To create the necessary conda environment, run
`conda env create -f bundle-adjust.yml`

# NOTE: This is a work in progress
We appreciate any kind of feedback. Open an issue or send a message, and we'll try our best to implement that feedback

# Instructions
The script `reconstruct_points.py` is where the reconstruction happens. It takes as input a path to a config file, e.g. `./configs/ibl.yaml`. To perform reconstruction run the code as follows:
``python reconstruct_points.py --config ./configs/ibl.yaml --output_dir ./broh/ --downsample 10 --reproj_thresh 6``
