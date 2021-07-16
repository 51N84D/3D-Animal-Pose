#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:17:10 2020

@author: danbiderman
"""

"""In: time indices; DLC tracked pts in 2D (cam1+cam2); DLC 3D tracked pts; recovered dict
we loop over time points, and present the images along with our 3D plot
figure should be presented using """
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import cv2
from scipy.spatial.transform import Rotation as R
from matplotlib import colors
from PIL import Image


# this function assumes that we're plotting the first n frames of our dictionaries
def video_recover_plots(
    noisy_pose_dict,
    recovered_pose_dict,
    stacked_images,
    DLC_sliced1,
    DLC_sliced2,
    reproj_list_of_dicts,
    save_name,
    title,
    n_frames,
    lim_abs,
):
    """seems to work.this version has the gridspec including
    what i learned - using p3.Axes3D along with regular ax.plot was helpful.
    in ax.plot I just use x,y,z, coordinates.
    there is no init function like in the 2d plots, we initialize the lines
    with the vales at the zeroth timepoint"""
    from matplotlib.animation import FuncAnimation
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(constrained_layout=True, figsize=(11, 6))

    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[:2, 0])
    import mpl_toolkits.mplot3d.axes3d as p3

    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax2 = fig.add_subplot(
        gs[:2, 1], projection="3d"
    )  # https://matplotlib.org/3.1.1/gallery/mplot3d/subplot3d.html

    # fig = plt.figure()
    ax2.view_init(elev=-82.0, azim=-72)  # specific for costa data, found manually

    #    ax.set_xlim3d([-2.0, 2.0])
    #    ax.set_xlabel('X')
    #
    #    ax.set_ylim3d([-2.0, 2.0])
    #    ax.set_ylabel('Y')
    #
    #    ax.set_zlim3d([-2.0, 2.0])
    #    ax.set_zlabel('Z')

    im_stacked = ax1.imshow(
        stacked_images[:, :, 0], cmap="gray"
    )  # all the lines below consider a SPECIFIC CROP
    (line_1,) = ax1.plot(
        DLC_sliced1[0 + 1, [0, 3, 6]] - 200,
        DLC_sliced1[0 + 1, [1, 4, 7]],
        "r*",
        markersize=10,
    )

    (line_1_reproj,) = ax1.plot(
        reproj_list_of_dicts[0]["x_coords"][:, 0 + 1] - 200,
        reproj_list_of_dicts[0]["y_coords"][:, 0 + 1],
        "b+",
        markersize=10,
    )
    #    scatt_1 = ax1.scatter(DLC_sliced1[0+1,[0,3,6]]-200, DLC_sliced1[0+1,[1,4,7]],
    #        color='red', marker='*', s=50)
    (line_2,) = ax1.plot(
        DLC_sliced2[0 + 1, [0, 3, 6]] - 220 + 300,
        DLC_sliced2[0 + 1, [1, 4, 7]] - 50,  # +300 due to stack
        "r*",
        markersize=10,
    )
    (line_2_reproj,) = ax1.plot(
        reproj_list_of_dicts[1]["x_coords"][:, 0 + 1] - 220 + 300,
        reproj_list_of_dicts[1]["y_coords"][:, 0 + 1] - 50,
        "b+",
        markersize=10,
    )

    line_v = ax1.axvline(x=300, ymin=0, ymax=300, color="white", linewidth=5)
    ax1.set_title("View 1             View 2", fontsize=16, weight="bold")
    ax1.axis("off")
    ax1.legend(loc="lower left", labels=["DLC 2D", "BA-reproj"])

    # init lines with true data, no init method.
    (line_noise,) = ax2.plot(
        noisy_pose_dict["x_coords"][:, 0],
        noisy_pose_dict["y_coords"][:, 0],
        noisy_pose_dict["z_coords"][:, 0],
        "ro-",
        lw=4,
        markersize=5,
        label="DLC 3D",
    )
    (line_recovered,) = ax2.plot(
        recovered_pose_dict["x_coords"][:, 0],
        recovered_pose_dict["y_coords"][:, 0],
        recovered_pose_dict["z_coords"][:, 0],
        "ko-",
        lw=4,
        markersize=5,
        label="BA_3D",
        alpha=0.5,
    )
    #    plt.legend(loc = "lower left")
    #    plt.title(title, fontsize = 16)

    # dan 2/28 can be avoided
    # new for costa data
    #     ax2.set_xlim3d([-2.0, 2.0])
    #     #ax.set_xticks([])
    #     ax2.set_xticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    #     ax2.set_xticklabels([])
    #     ax2.tick_params('x')
    #     ax2.set_xlabel('X', fontsize=12, labelpad = 0.1)
    #     ax2.set_ylim3d([-2.0, 2.0])
    #     ax2.set_yticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    #     ax2.set_yticklabels([])
    #     ax2.set_ylabel('Y', fontsize=12)
    #     ax2.set_zlim3d([-2.0, 2.0])
    #     ax2.set_zticks([-2.0, -1.0, 0.0, 1.0, 2.0])
    #     ax2.set_zticklabels([])
    #     ax2.set_zlabel('Z', fontsize=12)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.set_title("DLC Calib VS BA", fontsize=16, weight="bold", pad=18)

    # need to get images here
    def animate(
        i,
        stacked_images,
        im_stacked,
        DLC_sliced1,
        line_1,
        DLC_sliced2,
        line_2,
        reproj_list,
        line_1_reproj,
        line_2_reproj,
        line_v,
        noisy_pose_dict,
        line_noise,
        recovered_pose_dict,
        line_recovered,
    ):  # note, the loop has to see all these datasets

        im_stacked.set_data(
            stacked_images[:, :, i]
        )  # all the lines below consider a SPECIFIC CROP
        line_1.set_data(
            DLC_sliced1[i + 1, [0, 3, 6]] - 200, DLC_sliced1[i + 1, [1, 4, 7]]
        )
        line_1_reproj.set_data(
            reproj_list_of_dicts[0]["x_coords"][:, i + 1] - 200,
            reproj_list_of_dicts[0]["y_coords"][:, i + 1],
        )
        line_2.set_data(
            DLC_sliced2[i + 1, [0, 3, 6]] - 220 + 300,
            DLC_sliced2[i + 1, [1, 4, 7]] - 50,
        )
        line_2_reproj.set_data(
            reproj_list_of_dicts[1]["x_coords"][:, i + 1] - 220 + 300,
            reproj_list_of_dicts[1]["y_coords"][:, i + 1] - 50,
        )
        line_v = ax1.axvline(
            x=300, ymin=0, ymax=300, color="white", linewidth=5
        )  # .set_data(x=300, ymin = 0, ymax = 300)
        # noisy obs
        line_noise.set_data(
            noisy_pose_dict["x_coords"][:, i], noisy_pose_dict["y_coords"][:, i]
        )
        line_noise.set_3d_properties(noisy_pose_dict["z_coords"][:, i])

        # recovered
        line_recovered.set_data(
            recovered_pose_dict["x_coords"][:, i], recovered_pose_dict["y_coords"][:, i]
        )
        line_recovered.set_3d_properties(recovered_pose_dict["z_coords"][:, i])

        return (
            im_stacked,
            line_1,
            line_2,
            line_1_reproj,
            line_2_reproj,
            line_v,
            line_noise,
            line_recovered,
        )  # not sure the return is needed. check.

    anim = FuncAnimation(
        fig,
        animate,
        fargs=(
            stacked_images,
            im_stacked,
            DLC_sliced1,
            line_1,
            DLC_sliced2,
            line_2,
            reproj_list_of_dicts,
            line_1_reproj,
            line_2_reproj,
            line_v,
            noisy_pose_dict,
            line_noise,
            recovered_pose_dict,
            line_recovered,
        ),  # all inputs to loop
        frames=n_frames,
        interval=50,
        blit=False,
    )  # blit false important?

    # anim.save("recover.mp4")
    # anim.save(save_name + '.gif', writer='Imagekick') # works
    anim.save(save_name + ".gif", writer="pillow")  # pillow important


def plot_reproj_traces(points_2d, points_proj, frame_range, title, savename):
    """choose frame range according to num_frames_analyzed and
    num_body_parts and num_cameras. above, we take each body part coord, and stack
    it for all body parts and cameras."""
    plt.suptitle(title, fontsize=14)
    plt.subplot(211)
    plt.ylabel("x coord")
    plt.plot(points_2d[frame_range, 0], color="gray", linewidth=3)
    plt.plot(points_proj[frame_range, 0], color="red", linestyle="dashed", linewidth=1)
    plt.subplot(212)
    plt.ylabel("y coord")
    plt.plot(points_2d[frame_range, 1], color="gray", linewidth=3, label="DLC 2D")
    plt.plot(
        points_proj[frame_range, 1],
        color="red",
        linestyle="dashed",
        linewidth=1,
        label="re-projection",
    )
    plt.xlabel("frame num.")
    plt.legend(loc="lower right")
    plt.savefig(savename + ".png")


def plot_image_labels(
    image,
    coord_list_of_dicts,
    index,
    color_list,
    label_list=None,
    ax=None,
    top_img_height=0,
    pad=10,
):
    """ToDo: add multiple data sources, so instead of x_arr, y_arr have a list of dicts
    with x,y coords."""
    if len(image.shape) == 2:
        nrows, ncols = image.shape
    else:
        nrows, ncols = image.shape[0:2]

    assert len(color_list) == len(coord_list_of_dicts)

    ax.axis("off")
    ax.set_xlim(-pad, ncols + pad)  # pad x,y axes
    ax.set_ylim(nrows + pad, -pad)

    ax.imshow(image, "gray")
    for i in range(len(coord_list_of_dicts)):
        x_coord = np.copy(coord_list_of_dicts[i]["x_coords"][index, :])
        y_coord = np.copy(coord_list_of_dicts[i]["y_coords"][index, :])

        # indices 1 and 3 correspond to the bottom image.
        # Recall, we subtracted out the height of the top
        # image earlier.
        if i % 2 != 0:
            y_coord += top_img_height

        if label_list is not None:
            ax.scatter(
                x_coord, y_coord, color=color_list[i], label=label_list[i],
            )
        else:
            ax.scatter(x_coord, y_coord, color=color_list[i])


def plot_3d_points(coord_list_of_dicts, lims_dict, index, color_list, ax):
    """plot a single frame
    ToDo: specify plotting patams, allow for multiple inputs"""

    assert len(color_list) == len(coord_list_of_dicts)

    ax.set_xlim3d(lims_dict["x"])
    ax.set_ylim3d(lims_dict["y"])
    ax.set_zlim3d(lims_dict["z"])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    for i in range(len(coord_list_of_dicts)):
        ax.plot(
            coord_list_of_dicts[i]["x_coords"][index, :],
            coord_list_of_dicts[i]["y_coords"][index, :],
            coord_list_of_dicts[i]["z_coords"][index, :],
            linestyle="None",
            marker="o",
            color=color_list[i],
            markersize=4,
            label=None,
        )


def vector_plot(
    tvects,
    is_vect=True,
    orig=[0, 0, 0],
    # names=["x", "y", "z"],
    names=["", "", ""],
    cam_name="",
    colors=["red", "blue", "green"],
):
    """Plot vectors using plotly"""


    if is_vect:
        if not hasattr(orig[0], "__iter__"):
            coords = [[orig, np.sum([orig, v], axis=0)] for v in tvects]
        else:
            coords = [[o, np.sum([o, v], axis=0)] for o, v in zip(orig, tvects)]
    else:
        coords = tvects

    data = []
    for i, c in enumerate(coords):
        X1, Y1, Z1 = zip(c[0])
        X2, Y2, Z2 = zip(c[1])
        vector = go.Scatter3d(
            x=[X1[0], X2[0]],
            y=[Y1[0], Y2[0]],
            z=[Z1[0], Z2[0]],
            text=[cam_name, names[i]],
            marker=dict(size=[0, 5], color="blue"),
            line=dict(width=5, color=colors[i]),
            mode="lines+markers+text",
            name=names[i] + cam_name,
        )
        data.append(vector)

    return data
    # fig = go.Figure(data=data,layout=layout)
    # fig.show()


def draw_circles(
    img, points, point_colors=None, point_size=10, marker_type=None, thickness=-1
):

    if point_colors is not None:
        if not isinstance(point_colors, list):
            point_colors = [point_colors] * points.shape[0]
        assert len(point_colors) == points.shape[0]

    for i, point in enumerate(points):
        if point_colors is not None:
            color = [i * 255 for i in colors.to_rgb(point_colors[i])]
        else:
            color = (255, 255, 255)

        if marker_type is not None:
            assert marker_type in ("cross", "diamond", "square")
            if marker_type == "cross":
                marker = cv2.MARKER_CROSS

            cv2.drawMarker(
                img,
                (point[0], point[1]),
                markerType=marker,
                markerSize=point_size,
                thickness=2,
                color=(color[2], color[1], color[0], 0),
            )
        else:
            cv2.circle(
                img,
                (point[0], point[1]),
                point_size,
                (color[2], color[1], color[0]),
                thickness,
            )

    return img


def slope(x1, y1, x2, y2):
    ###finding slope
    if x2 != x1:
        return (y2 - y1) / (x2 - x1)
    else:
        return "NA"


def drawLine(image, x1, y1, x2, y2, color=None):

    m = slope(x1, y1, x2, y2)
    h, w = image.shape[:2]
    if m != "NA":
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with it
        ##starting point
        px = 0
        py = -(x1 - 0) * m + y1
        ##ending point
        qx = w
        qy = -(x2 - w) * m + y2
    else:
        ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px, py = x1, 0
        qx, qy = x1, h

    if color is not None:
        color = [i * 255 for i in colors.to_rgb(color)]
        image = cv2.line(
            image,
            (int(px), int(py)),
            (int(qx), int(qy)),
            (color[2], color[1], color[0]),
            2,
        )
    else:
        image = cv2.line(
            image, (int(px), int(py)), (int(qx), int(qy)), (255, 255, 255), 2
        )
    return image


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array(
        [
            [0, -vector[2], vector[1]],
            [vector[2], 0, -vector[0]],
            [-vector[1], vector[0], 0],
        ]
    )


def plot_cams_and_points(
    cam_group=None,
    points_3d=None,
    title=None,
    scene_lims=None,
    point_size=2,
    scene_aspect=None,
    scene_camera=None,
    show_plot=True,
    skeleton_bp=None,
    skeleton_lines=None,
    line_colors=None,
    point_colors=None,
    legend=False,
    font_size=15,
    plot_names=False,
    cam_names=None
):
    """
    Plots the coordinate systems of the cameras along with the 3D points
    """
    if points_3d is not None:
        scatter_data = [
            go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1],
                z=points_3d[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=point_colors),
            )
        ]

        data = scatter_data
    else:
        data = []

    if cam_group is not None:
        for i, cam in enumerate(cam_group.cameras):
            rot_vec = R.from_rotvec(cam.get_rotation())
            R_mat = rot_vec.as_matrix() * 0.2
            t = cam.get_translation()

            if cam_names is None:
                cam_name = f"cam_{i+1}"
            else:
                cam_name = cam_names[i]
            data += vector_plot(
                [R_mat[:, 0], R_mat[:, 1], R_mat[:, 2]], orig=t, cam_name=cam_name
            )
    if skeleton_bp is not None and skeleton_lines is not None:
        for i, line in enumerate(skeleton_lines):
            from_bp = line[0]
            to_bp = line[1]

            # Standard case
            if isinstance(from_bp, str):
                vec = [
                    skeleton_bp[from_bp][i] - skeleton_bp[to_bp][i] for i in range(3)
                ]

                if line_colors is not None:
                    color = [line_colors[i]]
                else:
                    color = ["gray"]

                cam_name = to_bp if plot_names else ""
                # names = [from_bp]
                names = [""]

                vec_data = vector_plot(
                    [vec],
                    orig=skeleton_bp[to_bp],
                    cam_name=cam_name,
                    names=names,
                    colors=color,
                )
                data += vec_data

            # Want to draw to midpoint
            elif isinstance(from_bp, tuple):
                # calculate midpoint
                vec_from_from = [
                    skeleton_bp[from_bp[0]][i] - skeleton_bp[from_bp[1]][i]
                    for i in range(3)
                ]

                mid = [
                    (skeleton_bp[from_bp[1]][j] + vec_from_from[j] * from_bp[-1])
                    for j in range(3)
                ]

                vec = [mid[i] - skeleton_bp[to_bp][i] for i in range(3)]
                if line_colors is not None:
                    color = [line_colors[i]]
                else:
                    color = ["green"]
                vec_data = vector_plot(
                    [vec],
                    orig=skeleton_bp[to_bp],
                    cam_name=to_bp,
                    names=[""],
                    colors=color,
                )
                data += vec_data

    if scene_lims is not None:
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(nticks=4, range=scene_lims[0], showticklabels=False),
                yaxis=dict(nticks=4, range=scene_lims[1], showticklabels=False),
                zaxis=dict(nticks=4, range=scene_lims[2], showticklabels=False),
            ),
        )
    else:
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0),)

    fig = go.FigureWidget(data=data, layout=layout)
    fig.update_traces(textfont_size=font_size)
    fig.update_layout(scene_aspectmode=scene_aspect)
    fig.update_layout(scene_camera=scene_camera)
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    if not legend:
        fig.update_layout(showlegend=False)

    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
    )

    if show_plot:
        fig.show()

    fig["layout"]["uirevision"] = "nothing"

    return fig


# Source: http://193.51.245.4/tutorials/convert_a_matplotlib_figure
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())
