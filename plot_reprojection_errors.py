import numpy as np
import matplotlib.pyplot as plt
import os
from utils.utils_IO import load_object
pickle_path = '/Volumes/sawtell-locker/C1/free/3d_reconstruction/20201104_Joao/reconstruction_results.pickle'
assert(os.path.isfile(pickle_path))
load_dict = load_object(pickle_path)

def plot_reprojection_traces(data_dict: dict, view_name: str, bp_name: str, coordinate_ind: int) -> None:
    coord_names = ["x", "y"]
    pred = data_dict["predictions"][view_name][bp_name][:, coordinate_ind]
    reproj = data_dict["reprojections"][view_name][bp_name][:, coordinate_ind]
    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=np.arange(len(pred)), y=pred, name='2D keypoints',
                             line=dict(color='firebrick', width=4)))
    fig.add_trace(go.Scatter(x=np.arange(len(reproj)), y=reproj, name='3D reprojections',
                             line=dict(color='royalblue', width=4)))
    fig.update_layout(
        title="Reprojected Keypoint Quality Control",
        xaxis_title="Frame #",
        yaxis_title="%s %s coord." % (bp_name, coord_names[coordinate_ind]),
        font=dict(
            size=18,
            color="black"
        )
    )

plot_reprojection_traces(load_dict, "main", "chin_base", 1) # plots and saves