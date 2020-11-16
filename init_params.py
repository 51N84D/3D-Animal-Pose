# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from anipose_BA import CameraGroup, Camera
from utils.utils_plotting import plot_cams_and_points, draw_circles
from utils.utils_IO import (
    refill_nan_array,
    ordered_arr_3d_to_dict,
    read_image,
    combine_images,
)
import plotly.io as pio
import plotly.graph_objs as go

# from preprocessing.preprocess_Sawtell import get_data
from preprocessing.preprocess_Sawtell_DLC import get_data

# from preprocessing.preprocess_IBL import get_data

import base64
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import json
from copy import deepcopy

from PIL import Image
import cv2
from tqdm import tqdm

pio.renderers.default = None


def get_cameras(
    img_widths, img_heights, focal_length, num_cameras, rvecs=None, tvecs=None
):
    cameras = []
    for i in range(num_cameras):
        # Manual initialization
        '''
        if i == 0:
            cam = Camera(rvec=[-np.pi / 2, 0, 0], tvec=[0, -1.94, 1.72])
        elif i == 1:
            cam = Camera(rvec=[0, 0, 0], tvec=[0, 0, 0])
        else:
            cam = Camera(rvec=[0, -np.pi / 2, 0], tvec=[1.86, 0, 1.72])

        if isinstance(focal_length, list):
            cam.set_focal_length(focal_length[i])
        else:
            cam.set_focal_length(focal_length)
        '''

        cam = Camera(rvec=[0, 0, 0], tvec=[0, 0, 0])

        # Set Offset
        cam_mat = cam.get_camera_matrix()
        cam_mat[0, 2] = img_widths[i] // 2
        cam_mat[1, 2] = img_heights[i] // 2
        cam.set_camera_matrix(cam_mat)
        cameras.append(cam)

    cam_group = CameraGroup(cameras=cameras)
    return cam_group


def refill_arr(points_2d, info_dict):

    clean_point_indices = info_dict["clean_point_indices"]
    num_points_all = info_dict["num_points_all"]
    num_frames = info_dict["num_frames"]
    all_point_indices = np.arange(num_points_all)

    nan_point_indices = np.asarray(
        [x for x in all_point_indices if x not in clean_point_indices]
    )

    # If points_2d is (num_views, num_points, 2):
    if len(points_2d.shape) == 3:
        filled_points_2d = np.empty((points_2d.shape[0], num_points_all, 2))
        filled_points_2d[:, clean_point_indices, :] = points_2d
        filled_points_2d[:, nan_point_indices, :] = np.nan
        filled_points_2d = np.reshape(
            filled_points_2d, (points_2d.shape[0], num_frames, -1, 2)
        )

    # Elif points2d is (num_points, 2) --> this happens if we consider each view separately
    elif len(points_2d.shape) == 2:
        filled_points_2d = np.empty((num_points_all, 2))
        filled_points_2d[clean_point_indices, :] = points_2d
        filled_points_2d[nan_point_indices, :] = np.nan
        filled_points_2d = np.reshape(filled_points_2d, (num_frames, -1, 2))
    return filled_points_2d


def reproject_points(points_3d, cam_group, info_dict):

    multivew_filled_points_2d = []

    for cam in cam_group.cameras:
        points_2d = np.squeeze(cam.project(points_3d))
        filled_points_2d = refill_arr(points_2d, info_dict)
        # Now, reshape back to (num_frames, num_points per frame, 2):
        multivew_filled_points_2d.append(filled_points_2d)

    multivew_filled_points_2d = np.asarray(multivew_filled_points_2d)
    return multivew_filled_points_2d


def get_skeleton_parts(slice_3d):
    skeleton_bp = {}
    skeleton_bp["head"] = tuple(slice_3d[0, :])
    skeleton_bp["chin_base"] = tuple(slice_3d[1, :])
    skeleton_bp["chin_mid"] = tuple(slice_3d[2, :])
    skeleton_bp["chin_end"] = tuple(slice_3d[3, :])
    skeleton_bp["mid"] = tuple(slice_3d[4, :])
    skeleton_bp["tail"] = tuple(slice_3d[5, :])
    skeleton_bp["caudal_d"] = tuple(slice_3d[6, :])
    skeleton_bp["caudal_v"] = tuple(slice_3d[7, :])

    skeleton_lines = [
        ("caudal_d", "tail"),
        ("caudal_v", "tail"),
        ("tail", "mid"),
        ("mid", "head"),
        ("head", "chin_base"),
        ("chin_base", "chin_mid"),
        ("chin_mid", "chin_end"),
    ]

    # return skeleton_bp, skeleton_lines
    return None, None


def get_reproject_images(points_2d_reproj, points_2d_og, path_images, i=0):
    # For each camera
    reproj_dir = Path("./reproject_images")
    reproj_dir.mkdir(exist_ok=True, parents=True)

    images = []
    for cam_num in range(len(path_images)):
        img_path = path_images[cam_num][i]
        img = plt.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        draw_circles(
            img, points_2d_og[cam_num, i, :, :].astype(np.int32), "red", point_size=5
        )
        draw_circles(
            img,
            points_2d_reproj[cam_num, i, :, :].astype(np.int32),
            "blue",
            point_size=5,
        )
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


def get_points_at_frame(points_3d, i=0):
    global info_dict
    BA_array_3d_back = refill_nan_array(points_3d, info_dict, dimension="3d")
    BA_dict = ordered_arr_3d_to_dict(BA_array_3d_back, info_dict)
    slice_3d = np.asarray(
        [BA_dict["x_coords"][i], BA_dict["y_coords"][i], BA_dict["z_coords"][i]]
    ).transpose()

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
experiment_data = get_data()
pts_array_2d = experiment_data["pts_array_2d"]
img_width = experiment_data["img_width"]
img_height = experiment_data["img_height"]
focal_length = experiment_data["focal_length"]
path_images = experiment_data["path_images"]
info_dict = experiment_data["info_dict"]
num_cameras = info_dict["num_cameras"]
bodypart_names = experiment_data["bodypart_names"]
num_frames = info_dict["num_frames"]

div_images = make_div_images([i[0] for i in path_images])

cam_group = get_cameras(img_width, img_height, focal_length, num_cameras)
cam_group_reset = cam_group

fig = plot_cams_and_points(
    cam_group=cam_group,
    points_3d=None,
    title="Camera Extrinsics",
    scene_aspect="data",
)

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
                        value=31,
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
        html.Div(dcc.Input(id="focal-len", type="text", value=str(focal_length[0]))),
        html.Div(id="focal-out"),
    ]
)
# ------------------------------------------------------------
# ------------------------------------------------------------

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
    for frame_i in tqdm(range(345, 2635)):
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
            )

        xe = 0
        ye = -1
        ze = -2
        scene_camera = dict(
            up=dict(x=0, y=-1, z=0),
            # center=dict(x=0, y=0.1, z=0),
            eye=dict(x=xe, y=ye, z=ze),
        )

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
    points_2d_reproj = reproject_points(POINTS_3D, cam_group, info_dict)
    points_2d_og = refill_arr(pts_array_2d, info_dict)

    for frame_i in tqdm(range(num_frames - 1)):
        plot_dir = Path("./Reprojections")
        plot_dir.mkdir(exist_ok=True, parents=True)

        reproj_images = get_reproject_images(
            points_2d_reproj, points_2d_og, path_images, i=frame_i
        )
        combined_proj = combine_images(reproj_images)
        cv2.imwrite(str(plot_dir / f"reproj_{frame_i}.jpg"), combined_proj)

    return ""


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
        f0, points_3d_init = cam_group.get_initial_error(pts_array_2d)
        POINTS_3D = points_3d_init

        points_2d_reproj = reproject_points(points_3d_init, cam_group, info_dict)
        points_2d_og = refill_arr(pts_array_2d, info_dict)

        reproj_paths = get_reproject_images(
            points_2d_reproj, points_2d_og, path_images, i=frame_i
        )

        div_images = make_div_images(reproj_paths, is_path=False)

        new_fig = plot_cams_and_points(
            cam_group=cam_group,
            points_3d=points_3d_init,
        )
        N_CLICKS_TRIANGULATE = n_clicks_triangulate

        slice_3d = get_points_at_frame(POINTS_3D, frame_i)
        skeleton_bp, skeleton_lines = get_skeleton_parts(slice_3d)

        skel_fig = plot_cams_and_points(
            cam_group=None,
            points_3d=slice_3d,
            point_size=5,
            title=f"3D Points at frame {frame_i}",
            skeleton_bp=skeleton_bp,
            skeleton_lines=skeleton_lines,
        )
    # This means we must bundle adjust
    elif n_clicks_bundle != N_CLICKS_BUNDLE:

        cam_group_reset = deepcopy(cam_group)

        res, points_3d_init = cam_group.bundle_adjust(pts_array_2d)
        POINTS_3D = points_3d_init

        points_2d_reproj = reproject_points(points_3d_init, cam_group, info_dict)
        points_2d_og = refill_arr(pts_array_2d, info_dict)

        reproj_paths = get_reproject_images(
            points_2d_reproj, points_2d_og, path_images, i=frame_i
        )

        div_images = make_div_images(reproj_paths, is_path=False)

        new_fig = plot_cams_and_points(
            cam_group=cam_group,
            points_3d=points_3d_init,
        )
        N_CLICKS_BUNDLE = n_clicks_bundle

        slice_3d = get_points_at_frame(POINTS_3D, frame_i)

        skeleton_bp, skeleton_lines = get_skeleton_parts(slice_3d)

        skel_fig = plot_cams_and_points(
            cam_group=None,
            points_3d=slice_3d,
            point_size=5,
            skeleton_bp=skeleton_bp,
            skeleton_lines=skeleton_lines,
        )

    else:
        if (
            N_CLICKS_TRIANGULATE > 0
            or N_CLICKS_BUNDLE > 0
            and POINTS_3D is not None
            and not changed_extrinsics
        ):
            points_2d_reproj = reproject_points(POINTS_3D, cam_group, info_dict)
            points_2d_og = refill_arr(pts_array_2d, info_dict)

            reproj_paths = get_reproject_images(
                points_2d_reproj, points_2d_og, path_images, i=frame_i
            )
            div_images = make_div_images(reproj_paths, is_path=False)

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
            )

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
# ------------------------------------------------------------


if __name__ == "__main__":
    app.run_server(debug=True)
