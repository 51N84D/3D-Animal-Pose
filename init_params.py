# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from utils.utils_plotting import plot_cams_and_points, draw_circles, drawLine, skew
from utils.utils_IO import (
    combine_images,
)
from utils.utils_BA import clean_nans
import plotly.io as pio
import plotly.graph_objs as go

from preprocessing.process_config import read_yaml

import base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
import json
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import cv2
from tqdm import tqdm
import argparse
pio.renderers.default = None


def get_args():
    parser = argparse.ArgumentParser(description="3d reconstruction")

    # dataset
    parser.add_argument("--config", type=str, default="./configs/sawtell.yaml")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int)
    parser.add_argument("--downsampling", type=int, default=1)
    parser.add_argument("--plot_epipolar", action="store_true")
    parser.add_argument("--dataset", default=None, help="str with name")
    parser.add_argument("--dataset_path", default=None, help="str with folder path")
    parser.add_argument("--skip_frame", default=None, type=int, help="frame to skip")
    parser.add_argument("--point_sizes", default=7, type=int, nargs='+', help="sizes of points in each view")
    parser.add_argument("--equal_size", action='store_true', help="equalize reprojection images")

    return parser.parse_args()


def get_F(pts_array_2d):
    points_2d = deepcopy(pts_array_2d)
    # Note, this returns pairwise F between view 1 to all other views
    points_2d = points_2d[:, ~np.isnan(points_2d).any(axis=2).any(axis=0), :]

    cam1_points = points_2d[0]
    F = []
    for i, single_view_points in enumerate(points_2d):
        # Skip first view
        if i == 0:
            continue
        curr_F, mask = cv2.findFundamentalMat(
            cam1_points, single_view_points, cv2.FM_LMEDS
        )
        F.append(curr_F)

    return F


def get_F_geometry():
    global cam_group

    # Get rotation vector
    camera_1 = cam_group.cameras[0]
    rot_vec_1 = R.from_rotvec(camera_1.get_rotation())
    R1 = rot_vec_1.as_matrix()
    t1 = camera_1.get_translation()
    K1 = camera_1.get_camera_matrix()

    F_all = []

    for camera_2 in cam_group.cameras[1:]:
        # Get rotation vector
        rot_vec_2 = R.from_rotvec(camera_2.get_rotation())
        R2 = rot_vec_2.as_matrix()
        # Get translation vector
        t2 = camera_2.get_translation()
        K2 = camera_2.get_camera_matrix()

        # --- Now compute relevant quantities for F estimation ------
        # Camera matrix basics: http://www.cs.cmu.edu/~16385/s17/Slides/11.1_Camera_matrix.pdf
        # Fundamental matrix computation: https://rb.gy/dd0nz2

        # Compute projection matrices
        P1 = np.matmul(K1, np.concatenate((R1, t1[:, np.newaxis]), axis=1))
        P2 = np.matmul(K2, np.concatenate((R2, t2[:, np.newaxis]), axis=1))

        # Get camera center (view 1)
        R1_inv = np.linalg.inv(R1) 
        C = np.matmul(-R1_inv, t1)
        C = np.append(C, 1)

        F = np.matmul(skew(np.matmul(P2, C)), np.matmul(P2, np.linalg.pinv(P1)))
        F_all.append(F)
    return F_all


def reproject_points(points_3d, cam_group):

    multivew_filled_points_2d = []

    for cam in cam_group.cameras:
        points_2d = np.squeeze(cam.project(points_3d))
        points_2d = points_2d.reshape(num_frames, num_bodyparts, 2)
        multivew_filled_points_2d.append(points_2d)

    multivew_filled_points_2d = np.asarray(multivew_filled_points_2d)
    return multivew_filled_points_2d


def get_skeleton_parts(slice_3d):

    global config

    skeleton_bp = {}

    for i in range(slice_3d.shape[0]):
        skeleton_bp[config.bp_names[i]] = tuple(slice_3d[i, :])
    skeleton_lines = config.skeleton

    return skeleton_bp, skeleton_lines


def filter_zero_view_points(points_2d_og):
    return np.all(np.any(np.isnan(points_2d_og), axis=-1), axis=0)


def get_reproject_images(points_2d_reproj, points_2d_og, path_images, i=0):
    global F
    global color_list
    global plot_epipolar
    global point_sizes

    reproj_dir = Path("./reproject_images")
    reproj_dir.mkdir(exist_ok=True, parents=True)
    points_filter = filter_zero_view_points(points_2d_og)

    images = []
    # For each camera
    for cam_num in range(len(path_images)):
        cam1_points_2d_og = points_2d_og[0, i, :, :]

        img_path = path_images[cam_num][i]
        img = plt.imread(img_path)

        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        if np.max(img) <= 1:
            img *= 255

        curr_points_2d_og = points_2d_og[cam_num, i, :, :]
        curr_points_2d_reproj = points_2d_reproj[cam_num, i, :, :]

        curr_points_2d_reproj[points_filter[i, :]] = np.nan

        # Get indices of interpolated points
        nonfilled_indices = np.any(np.isnan(curr_points_2d_og), axis=-1)
        points_nonfilled = np.empty_like(curr_points_2d_reproj)
        points_nonfilled[:] = np.nan
        points_filled = np.empty_like(curr_points_2d_reproj)
        points_filled[:] = np.nan
        num_nans = np.sum(nonfilled_indices)

        points_nonfilled[nonfilled_indices] = curr_points_2d_reproj[nonfilled_indices]
        points_filled[~nonfilled_indices] = curr_points_2d_reproj[~nonfilled_indices]

        draw_circles(img, curr_points_2d_og.astype(np.int32), color_list, point_size=point_sizes[cam_num])

        draw_circles(
            img,
            points_filled.astype(np.int32),
            color_list,
            point_size=point_sizes[cam_num] * 4,
            marker_type='cross'
        )

        draw_circles(
            img,
            points_nonfilled.astype(np.int32),
            color_list,
            point_size=point_sizes[cam_num],
            thickness=2
        )

        # Draw epipolar lines
        if cam_num != 0 and plot_epipolar and num_nans > 0:
            cam1_points_2d_og = cam1_points_2d_og.astype(np.int32)
            cam1_points_2d_og = cam1_points_2d_og[
                ~np.isnan(cam1_points_2d_og).any(axis=1)
            ]
            ones = np.ones(cam1_points_2d_og.shape[0])[:, np.newaxis]
            points_homog = np.concatenate((cam1_points_2d_og, ones), axis=-1)

            aug_F = np.tile(F[cam_num - 1], (points_homog.shape[0], 1, 1))
            lines = np.squeeze(np.matmul(aug_F, points_homog[:, :, np.newaxis]))
            for j, line_vec in enumerate(lines):
                if not nonfilled_indices[j]:
                    continue
                # Get x and y intercepts (on image) to plot
                # y = 0: x = -c/a
                x_intercept = int(-line_vec[2] / line_vec[0])
                # x = 0: y = -c/b
                y_intercept = int(-line_vec[2] / line_vec[1])
                img = drawLine(img, x_intercept, 0, 0, y_intercept, color=color_list[j])

        images.append(img)

    return images


def make_div_images(images=[], is_path=True):
    div_images = []
    # First set height:
    max_height = 400 / len(images)

    if is_path:
        for i in range(len(images)):
            image_filename = images[i]  # replace with your own image

            encoded_image = base64.b64encode(open(image_filename, "rb").read())

            div_images.append(
                html.Div(
                    html.Img(
                        src="data:image/png;base64,{}".format(encoded_image.decode()),
                        style={"max-height": f"{max_height}px", "float": "center"},
                    ),
                    style={"textAlign": "center"},
                )
            )

    else:
        for img in images:
            retval, buffer = cv2.imencode(".jpg", img)
            encoded_image = base64.b64encode(buffer)

            div_images.append(
                html.Div(
                    html.Img(
                        src="data:image/png;base64,{}".format(encoded_image.decode()),
                        style={"max-height": f"{max_height}px", "float": "center"},
                    ),
                    style={"textAlign": "center"},
                )
            )

    return div_images


def get_points_at_frame(points_3d, i=0, filtered=True):

    """
    if filtered:
        BA_array_3d_back = refill_nan_array(points_3d, info_dict, dimension="3d")
        BA_dict = ordered_arr_3d_to_dict(BA_array_3d_back, info_dict)
        slice_3d = np.asarray(
            [BA_dict["x_coords"][i], BA_dict["y_coords"][i], BA_dict["z_coords"][i]]
        ).transpose()
    """

    # else:
    points_3d = points_3d.reshape(num_frames, num_bodyparts, 3)
    slice_3d = points_3d[i, :, :]
    if filtered:
        filter_indices = filter_zero_view_points(pts_2d_joints)
        slice_3d[filter_indices[i]] = np.nan

    return slice_3d


def write_params(param_file="params.json"):
    # Write parameters to file
    param_dict = {}
    for i, cam in enumerate(cam_group.cameras):
        cam_num = i + 1
        cam_dict = {}
        cam_dict["translation"] = cam.get_translation().tolist()
        cam_dict["rotation"] = cam.get_rotation().tolist()
        cam_dict["camera_matrix"] = cam.get_camera_matrix().tolist()

        param_dict[f"cam_{cam_num}"] = cam_dict

    with open(param_file, "w") as outfile:
        json.dump(param_dict, outfile, indent=4)


def get_translation_sliders(num_cameras):
    trans_sliders = []
    trans_slider_vals = {}
    trans_slider_ids = ["x", "y", "z"]
    for i in range(3):
        trans_sliders.append(
            dcc.Slider(
                id=f"{trans_slider_ids[i]}-translate",
                min=-5,
                max=5,
                value=0,
                step=0.01,
                marks={
                    -1: "-1",
                    -2: "-2",
                    -3: "-3",
                    -4: "-4",
                    -5: "-5",
                    0: f"{trans_slider_ids[i]}-translate",
                    1: "1",
                    2: "2",
                    3: "3",
                    4: "4",
                    5: "5",
                },
            )
        )

    for i in range(num_cameras):
        trans_slider_vals[i] = {"x": 0, "y": 0, "z": 0}
    return trans_sliders, trans_slider_vals


def get_rotation_sliders(num_cameras):
    rot_sliders = []
    rot_slider_vals = {}
    rot_slider_ids = ["x", "y", "z"]
    for i in range(3):
        rot_sliders.append(
            dcc.Slider(
                id=f"{rot_slider_ids[i]}-rotate",
                min=-np.pi,
                max=np.pi,
                value=0,
                step=0.01,
                marks={
                    -np.pi: "-\u03C0",
                    -3 * np.pi / 4.0: "-3\u03C0/4",
                    -np.pi / 2.0: "-\u03C0/2",
                    -np.pi / 4.0: "-\u03C0/4",
                    0: f"{rot_slider_ids[i]}-rotate",
                    np.pi: "\u03C0",
                    3 * np.pi / 4.0: "3\u03C0/4",
                    np.pi / 2.0: "\u03C0/2",
                    np.pi / 4.0: "\u03C0/4",
                },
            )
        )

    for i in range(num_cameras):
        rot_slider_vals[i] = {"x": 0, "y": 0, "z": 0}
    return rot_sliders, rot_slider_vals


def get_dropdown(num_cameras):
    dropdown_options = []
    for i in range(num_cameras):
        dropdown_options.append({"label": f"Cam {i + 1}", "value": i + 1})
    return dropdown_options


# --------------Define global variables-------------------------

print("---------------- READING YAML--------------------")
args = get_args()

experiment_data = read_yaml(args.config, args.skip_frame)

config = experiment_data["config"]

pts_2d_joints = experiment_data["points_2d_joints"]
likelihoods = experiment_data["likelihoods"]

pts_2d = pts_2d_joints.reshape(
    pts_2d_joints.shape[0],
    pts_2d_joints.shape[1] * pts_2d_joints.shape[2],
    pts_2d_joints.shape[3],
)
pts_2d_filtered, clean_point_indices = clean_nans(pts_2d)
img_width = experiment_data["image_widths"]
img_height = experiment_data["image_heights"]

focal_length = experiment_data["focal_length"]
path_images = experiment_data["frame_paths"]
num_cameras = experiment_data["num_cams"]
if "bodypart_names" in experiment_data:
    bodypart_names = experiment_data["bodypart_names"]
num_bodyparts = experiment_data["num_bodyparts"]
num_frames = experiment_data["num_frames"]

ind_start = args.start_index
ind_end = args.end_index
if ind_end is None:
    ind_end = num_frames
downsampling = args.downsampling
plot_epipolar = args.plot_epipolar
point_sizes = args.point_sizes
if isinstance(point_sizes, int):
    point_sizes = [point_sizes] * num_cameras
equal_size = args.equal_size
print("------------------------------------------------")

color_list = config.color_list

div_images = make_div_images([i[0] for i in path_images])

cam_group = experiment_data["cam_group"]
cam_group_reset = cam_group

fig = plot_cams_and_points(
    cam_group=cam_group,
    points_3d=None,
    title="Camera Extrinsics",
    scene_aspect="data",
)

# Get fundamental matrix
# F = get_F(pts_2d)
F = get_F_geometry()

N_CLICKS_TRIANGULATE = 0
N_CLICKS_BUNDLE = 0
N_CLICKS_RESET = 0
POINTS_3D = None

trans_sliders, trans_slider_vals = get_translation_sliders(num_cameras)
rot_sliders, rot_slider_vals = get_rotation_sliders(num_cameras)
dropdown_options = get_dropdown(num_cameras)
# ------------------------------------------------------------
# ------------------------------------------------------------

# ----------------------Define App----------------------------
app = dash.Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1(children="Camera Init"),
        html.Div(
            children="""
        A GUI for specifying camera parameter initializations.
    """
        ),
        dcc.Dropdown(
            id="cam-dropdown",
            options=dropdown_options,
            value="2",
        ),
        html.Div(
            html.Button(
                "Bundle-Adjust",
                id="bundle-adjust-button",
                n_clicks=0,
                style={"float": "center"},
            ),
            style={"float": "right"},
        ),
        html.Div(
            html.Button(
                "Reset", id="reset-button", n_clicks=0, style={"float": "center"}
            ),
            style={"float": "right"},
        ),
        html.Div(
            [
                html.Div(
                    dcc.Slider(
                        id="frame-selection",
                        min=0,
                        max=len(path_images[0]) - 1,
                        step=1,
                        value=0,
                        vertical=True,
                    ),
                    style={
                        "float": "left",
                    },
                ),
                html.Div(
                    div_images,
                    id="images",
                    style={
                        "float": "left",
                        "marginTop": 20,
                        "marginBottom": 100,
                        "max-height": "500px",
                    },
                ),
                html.Div(
                    dcc.Graph(
                        id="skeleton-graph",
                        figure={
                            "data": None,
                            "layout": {
                                "uirevision": "nothing",
                                "title": "Points at Frame 0",
                            },
                        },
                    ),
                    style={
                        "float": "left",
                        "marginTop": 0,
                        "marginLeft": 100,
                        "width": 400,
                    },
                ),
                html.Div(
                    dcc.Graph(
                        id="main-graph",
                        figure={
                            "data": fig["data"],
                            "layout": {"uirevision": "nothing"},
                        },
                    ),
                    style={
                        "float": "right",
                        "marginTop": 0,
                        "width": 400,
                        "marginBottom": 500,
                    },
                ),
            ],
            className="row",
        ),
        html.Div(trans_sliders, style={"float": "bottom", "marginTop": 500}),
        html.Div(
            rot_sliders,
            style={"float": "bottom", "marginTop": 20},
        ),
        html.Button("Triangulate", id="triangulate-button", n_clicks=0),
        html.Div(
            html.Button("Write params to file", id="write-button", n_clicks=0),
            style={"float": "right"},
        ),
        html.Div(id="write-out", children="", style={"float": "right"}),
        html.Div(
            html.Button("Save plots", id="plot-button", n_clicks=0),
            style={"float": "right"},
        ),
        html.Div(id="plot-out", children="", style={"float": "right"}),
        html.Div(
            html.Button("Save reprojections", id="reproj-button", n_clicks=0),
            style={"float": "right"},
        ),
        html.Div(id="reproj-out", children="", style={"float": "right"}),
        html.Div(
            html.Button("Save traces", id="traces-button", n_clicks=0),
            style={"float": "right"},
        ),
        html.Div(id="traces-out", children="", style={"float": "right"}),
        html.Div(
            html.Button("Save points", id="points-button", n_clicks=0),
            style={"float": "right"},
        ),
        html.Div(id="points-out", children="", style={"float": "right"}),
        html.Div(dcc.Input(id="focal-len", type="text", value=str(focal_length[0]))),
        html.Div(id="focal-out"),
    ]
)

# --------------------Define Callbacks------------------------


@app.callback(
    Output("write-out", "children"),
    [Input("write-button", "n_clicks")],
)
def params_out(n_clicks):
    """Write parameters to file"""
    write_params()
    return ""


@app.callback(
    Output("plot-out", "children"),
    [Input("plot-button", "n_clicks")],
)
def save_skeleton(n_clicks):
    """Write parameters to file"""
    if n_clicks == 0:
        return
    global num_frames
    global ind_start
    global ind_end
    global downsampling

    xe = 0
    ye = -1
    ze = -2
    total_rot = 2 * np.pi
    # keep every n frames
    rotations = np.arange(
        start=0, stop=total_rot, step=total_rot / (ind_end - ind_start) * downsampling
    )

    for frame_i in tqdm(range(ind_start, ind_end)):
        if frame_i % downsampling != 0:
            continue
        plot_dir = Path("./3D_plots")
        plot_dir.mkdir(exist_ok=True, parents=True)
        slice_3d = get_points_at_frame(POINTS_3D, i=frame_i)

        if np.all(np.isnan(slice_3d)):
            slice_3d = np.zeros((1, 3))
            skel_fig = plot_cams_and_points(
                cam_group=None,
                points_3d=slice_3d,
                point_size=0,
                skeleton_bp=None,
                skeleton_lines=None,
                point_colors=color_list,
            )
        else:
            skeleton_bp, skeleton_lines = get_skeleton_parts(slice_3d)
            skel_fig = plot_cams_and_points(
                cam_group=None,
                points_3d=slice_3d,
                point_size=5,
                skeleton_bp=skeleton_bp,
                skeleton_lines=skeleton_lines,
                font_size=10,
                point_colors=color_list,
            )

        scene_camera = dict(
            up=dict(x=0, y=-1, z=0),
            # center=dict(x=0, y=0.1, z=0),
            eye=dict(x=xe, y=ye, z=ze),
        )
        cam_rot = R.from_rotvec([0, rotations[1], 0]).as_matrix()
        new_cam = np.matmul(cam_rot, np.asarray([xe, ye, ze]))
        xe, ye, ze = new_cam

        skel_fig_layout = deepcopy(fig)

        skel_fig_layout.update_layout(title={"text": f"3D Points at Frame {frame_i}"})

        padding = []
        scaling = 0.5
        padding.append(np.nanmax(POINTS_3D[:, 0]) - np.nanmin(POINTS_3D[:, 0]))
        padding.append(np.nanmax(POINTS_3D[:, 1]) - np.nanmin(POINTS_3D[:, 1]))
        padding.append(np.nanmax(POINTS_3D[:, 2]) - np.nanmin(POINTS_3D[:, 2]))
        padding = np.asarray(padding) * scaling

        skel_fig_layout.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[
                        np.nanmin(POINTS_3D[:, 0]),
                        np.nanmax(POINTS_3D[:, 0]) + padding[0],
                    ]
                ),
                yaxis=dict(
                    range=[
                        np.nanmin(POINTS_3D[:, 1]) - padding[1],
                        np.nanmax(POINTS_3D[:, 1]) + padding[1],
                    ]
                ),
                zaxis=dict(
                    range=[
                        np.nanmin(POINTS_3D[:, 2]),
                        np.nanmax(POINTS_3D[:, 2]),
                    ]
                ),
            )
        )

        skel_fig_layout["layout"]["uirevision"] = "nothing"
        skel_fig_layout["layout"]["scene"]["aspectmode"] = "cube"
        skel_fig_layout.update_layout(scene_camera=scene_camera)

        skel_fig["layout"] = skel_fig_layout["layout"]

        skel_fig.write_image(str(plot_dir / f"3dpoints_{frame_i}.png"))

    return ""


@app.callback(
    Output("reproj-out", "children"),
    [Input("reproj-button", "n_clicks")],
)
def save_reproj(n_clicks):
    """Write parameters to file"""
    if n_clicks == 0:
        return
    global num_frames
    global ind_start
    global ind_end
    global downsampling

    points_2d_reproj = reproject_points(POINTS_3D, cam_group)
    points_2d_og = pts_2d_joints

    for frame_i in tqdm(range(ind_start, ind_end)):
        if frame_i % downsampling != 0:
            continue
        plot_dir = Path("./reprojections")
        plot_dir.mkdir(exist_ok=True, parents=True)

        reproj_images = get_reproject_images(
            points_2d_reproj, points_2d_og, path_images, i=frame_i
        )
        combined_proj = combine_images(reproj_images, equal_size=equal_size)
        cv2.imwrite(str(plot_dir / f"reproj_{frame_i}.jpg"), combined_proj)

    return ""


@app.callback(
    Output("traces-out", "children"),
    [Input("traces-button", "n_clicks")],
)
def save_traces(n_clicks):
    if POINTS_3D is None:
        return None
    points_2d_reproj = reproject_points(POINTS_3D, cam_group)
    points_2d_og = pts_2d_joints

    # Shape is (num_views, num_frames, num_bodyparts, 2)
    # For each view, plot x vs. t

    figsize = (40, 20)

    tPlot, axes = plt.subplots(nrows=num_cameras, ncols=num_bodyparts, sharex=False, sharey=False, figsize=figsize)

    tPlot.suptitle('Traces', fontsize=20)

    plot_dir = Path("./traces")
    plot_dir.mkdir(exist_ok=True, parents=True)
    count = 0
    for view in range(points_2d_og.shape[0]):
        for bp in range(points_2d_og.shape[2]):
            color = color_list[bp]
            axes[view, bp].plot(points_2d_reproj[view, :, bp, 0], color='blue', label='reprojection')
            axes[view, bp].plot(points_2d_og[view, :, bp, 0], color='red', label='DLC/DGP output')
            axes[view, bp].plot((0, 0), label='bodypart color', color=color)
            axes[view, bp].set_ylabel('x')
            axes[view, bp].set_xlabel('frame')
            leg = axes[view, bp].legend()
            for line in leg.get_lines():
                line.set_linewidth(6.0)
            axes[view, bp].title.set_text(f'view {view + 1}, {config.bp_names[bp]}')
            count += 1

    tPlot.savefig(str(plot_dir / "point_traces.png"))

    # Now repeat for likelihoods
    if np.sum(~np.isnan(likelihoods)) > 0:
        tPlot, axes = plt.subplots(nrows=num_cameras, ncols=num_bodyparts, sharex=False, sharey=False, figsize=figsize)

        tPlot.suptitle('Confidences', fontsize=20)

        count = 0
        for view in range(likelihoods.shape[0]):
            for bp in range(likelihoods.shape[2]):
                color = color_list[bp]
                axes[view, bp].plot(likelihoods[view, :, bp], color='blue')
                axes[view, bp].plot((0, 0), label='bodypart color', color=color)
                axes[view, bp].set_ylabel('likelihood')
                axes[view, bp].set_xlabel('frame')
                leg = axes[view, bp].legend()
                for line in leg.get_lines():
                    line.set_linewidth(6.0)
                axes[view, bp].title.set_text(f'view {view + 1}, {config.bp_names[bp]}')
                count += 1

        tPlot.savefig(str(plot_dir / "point_confidences.png"))
        # cv2.imwrite(str(plot_dir / f"reproj_{frame_i}.jpg"), combined_proj)


@app.callback(
    Output("points-out", "children"),
    [Input("points-button", "n_clicks")],
)
def save_points(n_clicks):
    if POINTS_3D is None:
        return None
    points_dir = Path('./points').resolve()
    points_dir.mkdir(parents=True, exist_ok=True)
    np.save(points_dir / f'{Path(args.config).stem}_3d.npy', POINTS_3D)


'''
@app.callback(
    Output("focal-out", "children"),
    [Input("focal-len", "value")],
)
def update_focal(focal_length):
    """Update focal lengths from input"""
    global cam_group
    for cam in cam_group.cameras:
        cam.set_focal_length(float(focal_length))
    return f"Focal length {focal_length}"
'''


@app.callback(
    [
        dash.dependencies.Output("x-translate", "value"),
        dash.dependencies.Output("y-translate", "value"),
        dash.dependencies.Output("z-translate", "value"),
        dash.dependencies.Output("x-rotate", "value"),
        dash.dependencies.Output("y-rotate", "value"),
        dash.dependencies.Output("z-rotate", "value"),
    ],
    [
        dash.dependencies.Input("cam-dropdown", "value"),
    ],
)
def update_sliders(cam_val):
    """Update slider values to match selected camera"""
    x_val_trans = trans_slider_vals[int(cam_val) - 1]["x"]
    y_val_trans = trans_slider_vals[int(cam_val) - 1]["y"]
    z_val_trans = trans_slider_vals[int(cam_val) - 1]["z"]

    x_val_rot = rot_slider_vals[int(cam_val) - 1]["x"]
    y_val_rot = rot_slider_vals[int(cam_val) - 1]["y"]
    z_val_rot = rot_slider_vals[int(cam_val) - 1]["z"]

    return x_val_trans, y_val_trans, z_val_trans, x_val_rot, y_val_rot, z_val_rot


@app.callback(
    [
        dash.dependencies.Output("main-graph", "figure"),
        dash.dependencies.Output("skeleton-graph", "figure"),
        dash.dependencies.Output("images", "children"),
    ],
    [
        dash.dependencies.Input("cam-dropdown", "value"),
        dash.dependencies.Input("x-translate", "value"),
        dash.dependencies.Input("y-translate", "value"),
        dash.dependencies.Input("z-translate", "value"),
        dash.dependencies.Input("x-rotate", "value"),
        dash.dependencies.Input("y-rotate", "value"),
        dash.dependencies.Input("z-rotate", "value"),
        dash.dependencies.Input("triangulate-button", "n_clicks"),
        dash.dependencies.Input("bundle-adjust-button", "n_clicks"),
        dash.dependencies.Input("reset-button", "n_clicks"),
        dash.dependencies.Input("frame-selection", "value"),
    ],
)
def update_fig(
    cam_val,
    x_trans,
    y_trans,
    z_trans,
    x_rot,
    y_rot,
    z_rot,
    n_clicks_triangulate,
    n_clicks_bundle,
    n_clicks_reset,
    frame_i,
):
    """Main function to update figure and plot 3D points"""
    global fig
    global trans_slider_vals
    global N_CLICKS_TRIANGULATE
    global N_CLICKS_BUNDLE
    global N_CLICKS_RESET

    global pts_array_2d
    global cam_group
    global cam_group_reset

    global div_images

    global POINTS_3D
    global color_list
    global F

    cam_to_edit = cam_group.cameras[int(cam_val) - 1]
    curr_tvec = cam_to_edit.get_translation()
    curr_rvec = cam_to_edit.get_rotation()

    new_tvec = deepcopy(cam_to_edit.get_translation())
    new_rvec = deepcopy(cam_to_edit.get_rotation())

    if x_trans is not None:
        curr_tvec[0] = cam_to_edit.init_trans[0] + x_trans
    if y_trans is not None:
        curr_tvec[1] = cam_to_edit.init_trans[1] + y_trans
    if z_trans is not None:
        curr_tvec[2] = cam_to_edit.init_trans[2] + z_trans

    if x_rot is not None:
        curr_rvec[0] = cam_to_edit.init_rot[0] + x_rot
    if y_rot is not None:
        curr_rvec[1] = cam_to_edit.init_rot[1] + y_rot
    if z_rot is not None:
        curr_rvec[2] = cam_to_edit.init_rot[2] + z_rot

    if np.array_equal(curr_rvec, new_rvec) and np.array_equal(curr_tvec, new_tvec):
        changed_extrinsics = False
    else:
        changed_extrinsics = True

    cam_to_edit.set_translation(curr_tvec)
    cam_to_edit.set_rotation(curr_rvec)

    trans_slider_vals[int(cam_val) - 1]["x"] = x_trans
    trans_slider_vals[int(cam_val) - 1]["y"] = y_trans
    trans_slider_vals[int(cam_val) - 1]["z"] = z_trans

    rot_slider_vals[int(cam_val) - 1]["x"] = x_rot
    rot_slider_vals[int(cam_val) - 1]["y"] = y_rot
    rot_slider_vals[int(cam_val) - 1]["z"] = z_rot

    cam_group.cameras[int(cam_val) - 1] = cam_to_edit

    if n_clicks_reset != N_CLICKS_RESET:
        cam_group = cam_group_reset
        new_fig = plot_cams_and_points(
            cam_group=cam_group,
            points_3d=None,
        )
        N_CLICKS_RESET = n_clicks_reset

    # This means we must triangualte
    if n_clicks_triangulate != N_CLICKS_TRIANGULATE:
        # f0, points_3d_init = cam_group.get_initial_error(pts_array_2d)
        points_3d_init = cam_group.triangulate_progressive(pts_2d_joints)
        points_3d_init = np.reshape(
            points_3d_init,
            (
                points_3d_init.shape[0] * points_3d_init.shape[1],
                points_3d_init.shape[2],
            ),
        )

        POINTS_3D = points_3d_init
        N_CLICKS_TRIANGULATE = n_clicks_triangulate
        new_fig, skel_fig, div_images = plot_points(points_3d_init, frame_i)

    # This means we must bundle adjust
    elif n_clicks_bundle != N_CLICKS_BUNDLE:
        cam_group_reset = deepcopy(cam_group)

        res, points_3d_init = cam_group.bundle_adjust(pts_2d_filtered)
        F = get_F_geometry()

        # Triangulate after to reproject
        f0, points_3d_init = cam_group.get_initial_error(pts_2d)
        POINTS_3D = points_3d_init

        N_CLICKS_BUNDLE = n_clicks_bundle

        new_fig, skel_fig, div_images = plot_points(points_3d_init, frame_i)

    else:
        if (
            N_CLICKS_TRIANGULATE > 0
            or N_CLICKS_BUNDLE > 0
            and POINTS_3D is not None
            and not changed_extrinsics
        ):

            new_fig, skel_fig, div_images = plot_points(POINTS_3D, frame_i)

        else:
            div_images = make_div_images([i[frame_i] for i in path_images])

            new_fig = plot_cams_and_points(
                cam_group=cam_group,
                points_3d=None,
            )

            skel_fig = plot_cams_and_points(
                cam_group=None,
                points_3d=None,
            )

    fig = go.Figure(data=new_fig["data"], layout=fig["layout"])

    skel_fig_layout = deepcopy(fig)
    skel_fig_layout.update_layout(title={"text": f"3D Points at Frame {frame_i}"})
    if POINTS_3D is not None:
        padding = []
        scaling = [0.2, 0.2, 0.2]
        padding.append(
            (np.nanmax(POINTS_3D[:, 0]) - np.nanmin(POINTS_3D[:, 0])) * scaling[0]
        )
        padding.append(
            (np.nanmax(POINTS_3D[:, 1]) - np.nanmin(POINTS_3D[:, 1])) * scaling[1]
        )
        padding.append(
            (np.nanmax(POINTS_3D[:, 2]) - np.nanmin(POINTS_3D[:, 2])) * scaling[2]
        )
        padding = np.asarray(padding)

        skel_fig_layout.update_layout(
            scene=dict(
                xaxis=dict(
                    range=[
                        np.nanmin(POINTS_3D[:, 0]) - padding[0],
                        np.nanmax(POINTS_3D[:, 0]) + padding[0],
                    ]
                ),
                yaxis=dict(
                    range=[
                        np.nanmin(POINTS_3D[:, 1]) - padding[0],
                        np.nanmax(POINTS_3D[:, 1]) + padding[0],
                    ]
                ),
                zaxis=dict(
                    range=[
                        np.nanmin(POINTS_3D[:, 2]) - padding[0],
                        np.nanmax(POINTS_3D[:, 2]) + padding[0],
                    ]
                ),
            )
        )

    skel_fig_layout["layout"]["uirevision"] = "nothing"
    skel_fig_layout["layout"]["scene"]["aspectmode"] = "cube"

    return (
        {"data": fig["data"], "layout": fig["layout"]},
        {"data": skel_fig["data"], "layout": skel_fig_layout["layout"]},
        div_images,
    )

# ------------------------------------------------------------


def plot_points(points_3d, frame_i):
    global cam_group
    new_fig = plot_cams_and_points(
        cam_group=cam_group,
        points_3d=POINTS_3D,
    )
    slice_3d = get_points_at_frame(POINTS_3D, frame_i)
    skeleton_bp, skeleton_lines = get_skeleton_parts(slice_3d)

    skel_fig = plot_cams_and_points(
        cam_group=None,
        points_3d=slice_3d,
        point_size=5,
        scene_aspect="cube",
        skeleton_bp=skeleton_bp,
        skeleton_lines=skeleton_lines,
        point_colors=color_list,
    )
    points_2d_reproj = reproject_points(POINTS_3D, cam_group)
    points_2d_og = pts_2d_joints

    reproj_paths = get_reproject_images(
        points_2d_reproj, points_2d_og, path_images, i=frame_i
    )
    div_images = make_div_images(reproj_paths, is_path=False)
    return new_fig, skel_fig, div_images


if __name__ == "__main__":
    app.run_server(debug=True)
