# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
from anipose_BA import CameraGroup, Camera
from utils.utils_plotting import plot_cams_and_points
import plotly.io as pio
import plotly.graph_objs as go

pio.renderers.default = None  # "vscode"


def get_cameras(img_size=(256, 256)):
    P_X_1 = P_X_2 = img_size[0] // 2
    P_Y_1 = P_Y_2 = img_size[1] // 2

    # --------CAMERA 1------------
    # Initialize camera 1
    camera_1 = Camera(rvec=[0, 0, 0], tvec=[0, 0, 0])

    cam1_init_params = np.abs(np.random.rand(8))
    # Set rotations [0:3] and translation [3:6] to 0
    cam1_init_params[0:6] = 0
    # Initialize focal length to image width
    cam1_init_params[6] = 1000
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
    cam2_init_params[6] = 1000
    cam2_init_params[7] = 0.0
    camera_2.set_params(cam2_init_params)
    camera_2_mat = camera_2.get_camera_matrix()
    camera_2_mat[0, 2] = P_X_2
    camera_2_mat[1, 2] = P_Y_2
    camera_2.set_camera_matrix(camera_2_mat)

    # Group cameras
    cam_group = CameraGroup(cameras=[camera_1, camera_2])
    return cam_group


points_3d = np.random.random((1000, 3))
cam_group = get_cameras()

fig = plot_cams_and_points(
    cam_group=cam_group,
    points_3d=None,
    title="Camera Extrinsics",
    scene_aspect="data",
)

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
                -np.pi / 2.0: "-\u03C0/2",
                -np.pi / 4.0: "-\u03C0/4",
                0: f"{rot_slider_ids[i]}-rotate",
                np.pi: "\u03C0",
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
    ]
)


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
    dash.dependencies.Output("main-graph", "figure"),
    [
        dash.dependencies.Input("cam-dropdown", "value"),
        dash.dependencies.Input("x-translate", "value"),
        dash.dependencies.Input("y-translate", "value"),
        dash.dependencies.Input("z-translate", "value"),
        dash.dependencies.Input("x-rotate", "value"),
        dash.dependencies.Input("y-rotate", "value"),
        dash.dependencies.Input("z-rotate", "value"),
    ],
)
def update_output(cam_val, x_trans, y_trans, z_trans, x_rot, y_rot, z_rot):
    global fig
    global trans_slider_vals

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

    new_fig = plot_cams_and_points(
        cam_group=cam_group,
        points_3d=None,
    )

    fig = go.Figure(data=new_fig["data"], layout=fig["layout"])

    return {"data": fig["data"], "layout": fig["layout"]}


print('BROH: ', cam_group.cameras[-1].get_translation())

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
