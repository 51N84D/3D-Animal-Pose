# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from anipose_BA import CameraGroup, Camera
from utils.utils_plotting import plot_cams_and_points, plot_image_labels
from utils.utils_IO import reproject_3d_points, read_image
import plotly.io as pio
import plotly.graph_objs as go
from preprocessing.preprocess_IBL import get_data
import base64
import matplotlib.pyplot as plt
from pathlib import Path
import json

pio.renderers.default = None


def get_cameras(img_width, img_height, focal_length, rvecs=None, tvecs=None):
    P_X_1 = P_X_2 = img_width[0] // 2
    P_Y_1 = P_Y_2 = img_height[0] // 2

    # --------CAMERA 1------------
    # Initialize camera 1
    camera_1 = Camera(rvec=[0, 0, 0], tvec=[0, 0, 0])

    cam1_init_params = np.abs(np.random.rand(8))
    # Set rotations [0:3] and translation [3:6] to 0
    cam1_init_params[0:6] = 0
    # Initialize focal length to image width
    cam1_init_params[6] = focal_length
    # Initialize distortion to 0
    cam1_init_params[7] = 0.0
    # Set parameters
    camera_1.set_params(cam1_init_params)
    camera_1_mat = camera_1.get_camera_matrix()
    camera_1_mat[0, 2] = P_X_1
    camera_1_mat[1, 2] = P_Y_1
    camera_1.set_camera_matrix(camera_1_mat)

    # --------CAMERA 2------------
    # Set rotation vector w.r.t. camera 1
    # roration around y axis only, about 120 deg (2.0127 rad) from Guido's CAD
    # rvec2 = np.array([0, 2.0127, 0])
    rvec2 = np.array([0, 0, 0])

    # Set translation vector w.r.t. camera 1, using CAD drawing [mm];
    # cameras are 292.8 mm apart;
    # distance vector pointing from cam1 to the other cam:
    # tvec2 = [-1.5664, 0, 2.4738]
    tvec2 = [0, 0, 0]

    # Initialize camera 2
    camera_2 = Camera(rvec=rvec2, tvec=tvec2)
    # Set offset
    camera_2.set_size((P_X_2, P_Y_2))

    cam2_init_params = np.abs(np.random.rand(8))
    cam2_init_params[0:3] = rvec2
    cam2_init_params[3:6] = tvec2
    cam2_init_params[6] = focal_length
    cam2_init_params[7] = 0.0
    camera_2.set_params(cam2_init_params)
    camera_2_mat = camera_2.get_camera_matrix()
    camera_2_mat[0, 2] = P_X_2
    camera_2_mat[1, 2] = P_Y_2
    camera_2.set_camera_matrix(camera_2_mat)

    # Group cameras
    cam_group = CameraGroup(cameras=[camera_1, camera_2])
    return cam_group


def get_reproject_images(joined_list_2d, path_images, i=0):
    fig, ax = plt.subplots()
    img_1 = read_image(path_images[0][i], flip=False)
    img_2 = read_image(path_images[1][i], flip=False)
    img = np.concatenate((img_1, img_2), axis=0)
    color_list_2d = ["red", "red", "blue", "blue"]
    plot_image_labels(
        img, joined_list_2d, i, color_list_2d, ax=ax, top_img_height=img_height[0]
    )
    reproj_dir = Path("./reproject_images")
    reproj_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(reproj_dir / f"img_{i}.png")
    plt.clf()
    return reproj_dir / f"img_{i}.png"


def make_div_images(image_path=[]):
    div_images = []
    for i in range(len(image_path)):
        image_filename = image_path[i]  # replace with your own image
        encoded_image = base64.b64encode(open(image_filename, "rb").read())
        div_images.append(
            html.Div(
                html.Img(
                    src="data:image/png;base64,{}".format(encoded_image.decode()),
                ),
                style={"textAlign": "center"},
            )
        )
    return div_images


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


experiment_data = get_data()
pts_array_2d = experiment_data["pts_array_2d"]
img_width = experiment_data["img_width"]
img_height = experiment_data["img_height"]
focal_length = (
    experiment_data["focal_length_mm"] * experiment_data["img_width"][0]
) / experiment_data["sensor_size"]
path_images = experiment_data["path_images"]
info_dict = experiment_data["info_dict"]

div_images = make_div_images([path_images[0][0], path_images[1][0]])

cam_group = get_cameras(img_width, img_height, focal_length)

fig = plot_cams_and_points(
    cam_group=cam_group,
    points_3d=None,
    title="Camera Extrinsics",
    scene_aspect="data",
)

N_CLICKS = 0

app = dash.Dash(__name__)

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

for i in range(len(cam_group.cameras)):
    trans_slider_vals[i] = {"x": 0, "y": 0, "z": 0}

for i in range(len(cam_group.cameras)):
    rot_slider_vals[i] = {"x": 0, "y": 0, "z": 0}

dropdown_options = []
for i in range(len(cam_group.cameras)):
    dropdown_options.append({"label": f"Cam {i + 1}", "value": i + 1})

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
            dcc.Graph(
                id="main-graph",
                figure={"data": fig["data"], "layout": {"uirevision": "nothing"}},
            ),
            style={"float": "center", "marginTop": 20},
        ),
        html.Div(trans_sliders, style={"float": "bottom", "marginTop": 20}),
        html.Div(
            rot_sliders,
            style={"float": "bottom", "marginTop": 20},
        ),
        html.Button("Triangulate", id="triangulate-button", n_clicks=0),
        html.Div(
            html.Button("Write params to file", id="write-button", n_clicks=0),
            style={"float": "right"},
        ),
        html.Div(dcc.Input(id="focal-len", type="text", value=str(focal_length))),
        html.Div(id="focal-out"),
        html.Div(id="write-out", children="", style={"float": "right"}),

        html.Div(div_images, id="images", style={"float": "center", "marginTop": 100}),


    ]
)


@app.callback(
    Output("write-out", "children"),
    [Input("write-button", "n_clicks")],
)
def params_out(n_clicks):
    write_params()
    return "Writing params to file: params.json"


@app.callback(
    Output("focal-out", "children"),
    [Input("focal-len", "value")],
)
def update_output(focal_length):
    global cam_group
    for cam in cam_group.cameras:
        cam.set_focal_length(float(focal_length))
    return f"Focal length {focal_length}"


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
    ],
)
def update_fig(cam_val, x_trans, y_trans, z_trans, x_rot, y_rot, z_rot, n_clicks):
    global fig
    global trans_slider_vals
    global N_CLICKS
    global pts_array_2d
    global cam_group
    global div_images

    cam_to_edit = cam_group.cameras[int(cam_val) - 1]
    curr_tvec = cam_to_edit.get_translation()
    curr_rvec = cam_to_edit.get_rotation()

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

    cam_to_edit.set_translation(curr_tvec)
    cam_to_edit.set_rotation(curr_rvec)

    trans_slider_vals[int(cam_val) - 1]["x"] = x_trans
    trans_slider_vals[int(cam_val) - 1]["y"] = y_trans
    trans_slider_vals[int(cam_val) - 1]["z"] = z_trans

    rot_slider_vals[int(cam_val) - 1]["x"] = x_rot
    rot_slider_vals[int(cam_val) - 1]["y"] = y_rot
    rot_slider_vals[int(cam_val) - 1]["z"] = z_rot

    cam_group.cameras[int(cam_val) - 1] = cam_to_edit

    if n_clicks != N_CLICKS:
        f0, points_3d_init = cam_group.get_initial_error(pts_array_2d)
        joined_list_2d = reproject_3d_points(
            points_3d_init, info_dict, pts_array_2d, cam_group
        )
        reproj_path = get_reproject_images(joined_list_2d, path_images)
        div_images = make_div_images([reproj_path])

        new_fig = plot_cams_and_points(
            cam_group=cam_group,
            points_3d=points_3d_init,
        )
        N_CLICKS = n_clicks

    else:
        new_fig = plot_cams_and_points(
            cam_group=cam_group,
            points_3d=None,
        )

    fig = go.Figure(data=new_fig["data"], layout=fig["layout"])

    return {"data": fig["data"], "layout": fig["layout"]}, div_images


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
