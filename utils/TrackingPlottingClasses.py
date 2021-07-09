import os
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

def set_or_open_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print("Opened a new folder at: {}".format(folder_path))
    else:
        print("The folder already exists at: {}".format(folder_path))
    return Path(folder_path) # a PosixPath object

# TODO: should go into some util class?
def pts_to_coords(pt1, pt2):
    # type: (np.ndarray, np.ndarray) -> dict
    '''converts [(x_1, y_1), (x_2, y_2)] --> ([x_1, x_2), (y_1, y_2)]
    supports a third coordinate if it appears.'''
    assert (pt1.shape == pt2.shape)
    keys = ["x", "y"]
    if pt1.shape[0] == 3:  # append a third coordinate
        keys.append("z")
    coord_dict = {}
    for i in range(pt1.shape[0]):
        coord_dict[keys[i]] = np.array([pt1[i], pt2[i]])
    return coord_dict

class Tracking_Video_Generator:
    """Parameters
    ----------
    save_file
    images_list : dict
        keys are 'view_name'. each view will include an array of number of images.
    points : dict of dicts of dicts: keys: [data_source][view_name][bpname]
        each dict has keys 'left', 'right', values are themselves dicts with keys of marker names
        and vals of marker values, i.e. `points['left']['paw_l'].shape = (n_t, 2)`
    titles : list
    colors : list
    framerate : float
        framerate of video
    height : float
        height of movie in inches
    frame_idxs_left_view : array-like, optional
        if present, frame index of left view is printed in upper corner of left view
    frame_idxs_right_view : array-like, optional
        if present, frame index of right view is printed in upper corner of right view
    """

    def __init__(self, images, points, skeleton_dict = None, num_plot_rows=1,
                 marker_size=50, figsize=(10, 5), marker_list=None, image_folder=None):
        """a class for generating videos"""
        self.images = images
        # self.images = self.check_images_list() # get rid of fourth dimension. TODO: not sure if best practice.
        self.points = points
        self.fig, self.ax = plt.subplots(num_plot_rows,
                                         int(len(list(self.images.keys())) / num_plot_rows),
                                         figsize=figsize)
        self.marker_size = marker_size
        self.ax = self.ax.flatten()  # so that you loop using just one index.
        self.data_sources = list(self.points.keys())
        self.views = list(self.images.keys())  # assuming identical keys for each data_source
        self.bp_names = list(self.points[self.data_sources[0]][self.views[0]].keys())  # "" data_source and view
        self.num_frames = images[self.views[0]].shape[0]
        self.image_type = 'jpg' #ToDo: specify some json with config?
        self.image_folder = set_or_open_folder(image_folder)
        self.skeleton_dict = skeleton_dict

        # set marker shapes for each data_source e.g., ["data", "predictions", "predictions2"...]
        if marker_list is not None:
            self.marker_list = marker_list
        else:  # pick the first few markers (not necassarily the best ones...)
            self.marker_list = list(matplotlib.markers.MarkerStyle.markers.keys())[:len(self.data_sources)]

        ## set a different color for each bodypart
        ## TODO: supporting up to 10 bodyparts with TABLEAU_COLORS.
        ## consider defining a different color map. the problem was plotting ellipses with RGB values IIRC
        #self.color_list = list(matplotlib.colors.TABLEAU_COLORS.keys())
        ## slice the list, assuming that self.points has the same number of bodyparts for each data_source
        #self.color_list = self.color_list[:len(self.bp_names)]
        self.color_list = cm.rainbow(np.linspace(0, 1, len(self.bp_names)))

    def show_images(self, frame_idx):
        # loop over views and display images on separate axes.
        for ax, view in zip(self.ax, self.views):
            ax.imshow(self.images[view][frame_idx, :, :], "gray")

    def check_images_list(self):
        # TODO: RE-WRITE or REMOVE
        # modify images_list if its 4-dimensional containing ones in the final dim
        for i in range(len(self.images_list)):
            if len(np.shape(self.images_list[i])) == 4:
                self.images_list[i] = self.images_list[i].reshape(self.images_list[i].shape[:3])
                print("reshaping images_list[{}]: new shape is: {}".format(i, self.images_list[i].shape))
        return self.images_list

    def scatter_points(self, frame_idx):
        '''loop over views and data sources. plot bodyparts one by one with their color.
        TODO: should add complications such as: do we plot the means or not?
        how to mark bodyparts missing from 1/3 views?
        what happens if we want to not plot a bodypart?'''

        # TODO: do we like the hierarchy of data_source, view, bp? matt suggests switching view and data_source. we may have a different set of data_source per view
        for ax, view in zip(self.ax, self.views):  # loop over views
            for data_source_ind, data_source in enumerate(
                    self.data_sources):  # loop over data sources (data, preds, ...)
                for bp_ind, bp_name in enumerate(self.bp_names):  # loop over bodyparts within a datasource
                    ax.scatter(self.points[data_source][view][bp_name][frame_idx, 0],
                               self.points[data_source][view][bp_name][frame_idx, 1],
                               color=self.color_list[bp_ind],
                               marker=self.marker_list[data_source_ind],
                               s=self.marker_size)

    def vid_from_images(self, video_name=None):
        # TODO: some of these, like -crf 25, or frame rate, can be control parameters.
        # TODO: switch to ffmpeg-python, see https://github.com/kkroening/ffmpeg-python/tree/master/examples
        string = " "
        cmd = string.join(["ffmpeg -r 2 -i",
                           os.path.join(str(self.image_folder),
                                        'im_%04d.{}'.format(self.image_type)),
                           "-vcodec libx264 -crf 25",
                           video_name])
        print(cmd)
        # cmd = "ffmpeg -r 2 -i images/IBL/im_%04d.jpg -vcodec libx264 -crf 25 ibl_pca3_hard_frames_maha_9.mp4"
        os.system(cmd)

    def clear_axes(self):
        # loop over axes and clear them
        for ax in self.ax:
            ax.cla()

    # TODO: the method below repeats in both classes. how can we abstract away the copies?
    # TODO: test. check how it runs in the case of fish versus IBL.
    def plot_skeleton(self, frame_idx, data_source):
        assert(self.skeleton_dict is not None)
        for view_ind, view_name in enumerate(self.points[datasource].keys()): # loop over view names
            if len(list(self.images.keys())) == 1: # handling the case that we have a single ax, single image.
                ax = self.ax
            else:
                ax = self.ax[view_ind]
            for idx, name in enumerate(self.skeleton_dict["name"]):
                if self.skeleton_dict["parent"][idx] is not None:  # if bodypart has a parent
                    name = self.points[data_source][view_name][name][frame_idx, :]
                    parent = self.points[data_source][view_name][self.skeleton_dict["parent"][idx]][frame_idx, :]
                    coord_dict = pts_to_coords(name, parent)
                    ax.plot(coord_dict["x"],
                                 coord_dict["y"],
                                 'gray',
                                 marker=None)


    def __call__(self, video_name=None):
        for fr in tqdm(range(self.num_frames)):
            self.clear_axes()
            self.show_images(fr)
            self.scatter_points(fr)
            self.fig.savefig(self.image_folder / ('im_%04d.%s' % (fr, self.image_type)))  # was png
            # save a png / jpg to folder given in self.__init__()
        if video_name is not None:
            self.vid_from_images(str(video_name))
        # TODO: erase image_folder



from mpl_toolkits.mplot3d import Axes3D

class three_D_skeleton_plotter:
    """Parameters
    ----------
    save_file
    images_list : dict
        keys are 'view_name'. each view will include an array of number of images.
    points : dict of dicts: keys: [data_source][bpname]
    i.e. `points['BA']['paw_l'].shape = (n_t, 3)`
    ax: ax = fig.add_subplot(projection = '3d') # has to be 3D ax. https://matplotlib.org/3.1.1/gallery/mplot3d/subplot3d.html
    titles : list
    colors : list
    framerate : float
        framerate of video
    height : float
        height of movie in inches
    frame_idxs_left_view : array-like, optional
        if present, frame index of left view is printed in upper corner of left view
    frame_idxs_right_view : array-like, optional
        if present, frame index of right view is printed in upper corner of right view
    """

    def __init__(self, points, skeleton_dict=None, marker_size=50, figsize=(5, 5), ax=None,
                 marker_list=None, init_elev=30, init_azim=90,
                 image_folder=None):
        """a class for generating videos"""
        # self.images = self.check_images_list() # get rid of fourth dimension. TODO: not sure if best practice.
        self.points = points
        if ax == None:
            self.fig = plt.figure(figsize=figsize)
            self.ax = self.fig.add_subplot(1, 1, 1, projection='3d')
        else:
            self.ax = ax

        self.init_elev = init_elev
        self.init_azim = init_azim
        self.ax.view_init(elev=self.init_elev, azim=self.init_azim)
        self.marker_size = marker_size
        self.data_sources = list(self.points.keys())
        self.bp_names = list(self.points[self.data_sources[0]].keys())  # "" data_source and view
        self.num_frames = self.points[self.data_sources[0]][self.bp_names[0]].shape[0]
        self.image_type = 'jpg'
        if image_folder is not None:
            self.image_folder = set_or_open_folder(image_folder)
        self.skeleton_dict = skeleton_dict

        # set marker shapes for each data_source e.g., ["data", "predictions", "predictions2"...]
        if marker_list is not None:
            self.marker_list = marker_list
        else:  # pick the first few markers (not necassarily the best ones...)
            self.marker_list = list(matplotlib.markers.MarkerStyle.markers.keys())[:len(self.data_sources)]

        # # set a different color for each bodypart
        # # TODO: supporting up to 10 bodyparts with TABLEAU_COLORS.
        # # consider defining a different color map. the problem was plotting ellipses with RGB values IIRC
        # self.color_list = list(matplotlib.colors.TABLEAU_COLORS.keys())
        # # slice the list, assuming that self.points has the same number of bodyparts for each data_source
        # self.color_list = self.color_list[:len(self.bp_names)]
        self.color_list = cm.rainbow(np.linspace(0, 1, len(self.bp_names))) # TODO: expect problem with contour

        self.lims_dict = self.calc_xyz_lims_dict(self.collapse_dict())  # TODO: 20 is hard coded, no need for that
        self.set_xyz_lims()
        self.set_labels()

    def calc_xyz_lims_dict(self, collapsed_arr):
        assert isinstance(collapsed_arr, np.ndarray)
        lims = {}
        mins = np.nanmin(collapsed_arr, axis=0)  # - pad
        maxs = np.nanmax(collapsed_arr, axis=0)  # + pad
        mins, maxs = self.pad_lims(mins, maxs)
        keys = ["x", "y", "z"]
        for i, k in enumerate(keys):
            lims[k] = [mins[i], maxs[i]]
        return lims

    @staticmethod
    def pad_lims(mins, maxs, proportion_pad=0.15):
        pad = proportion_pad * np.max(np.stack([np.abs(mins), np.abs(maxs)], axis=0), axis=0)
        return (mins - pad, maxs + pad)

    def set_labels(self, labels=['X', 'Y', 'Z']):
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.set_zlabel(labels[2])

    def set_xyz_lims(self):
        self.ax.set_xlim3d(self.lims_dict["x"])
        self.ax.set_ylim3d(self.lims_dict["y"])
        self.ax.set_zlim3d(self.lims_dict["z"])

    def collapse_dict(self, data_source=None):
        # TODO: right now supports one data_source.
        if data_source is None:
            data_source = list(self.points.keys())[0]
        collapsed_list = []
        for key in self.points[data_source].keys():
            collapsed_list.append(self.points[data_source][key])
        return np.asarray(collapsed_list).reshape(-1, 3)

    def scatter_points(self, frame_idx):
        '''loop over views and data sources. plot bodyparts one by one with their color.
        TODO: should add complications such as: do we plot the means or not?
        how to mark bodyparts missing from 1/3 views?
        TODO: loop over body parts could be avoided. colors could be specified in a list. https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
        what happens if we want to not plot a bodypart?'''

        # TODO: do we like the hierarchy of data_source, view, bp? matt suggests switching view and data_source. we may have a different set of data_source per view
        for data_source_ind, data_source in enumerate(self.data_sources):  # loop over data sources (data, preds, ...)
            for bp_ind, bp_name in enumerate(self.bp_names):  # loop over bodyparts within a datasource
                self.ax.scatter(self.points[data_source][bp_name][frame_idx, 0],
                                self.points[data_source][bp_name][frame_idx, 1],
                                self.points[data_source][bp_name][frame_idx, 2],
                                color=self.color_list[bp_ind],
                                marker=self.marker_list[data_source_ind],
                                s=self.marker_size)

    def set_curr_elev_azim(self, elev=None, azim=None):
        if elev is None:
            elev = self.init_elev
        if azim is None:
            azim = self.init_azim
        self.ax.elev = elev  # see https://stackoverflow.com/questions/12904912/how-to-set-camera-position-for-3d-plots-using-python-matplotlib
        self.ax.azim = azim

    @staticmethod
    def pts_to_coords(pt1, pt2):
        '''converts [(x_1, y_1), (x_2, y_2)] --> ([x_1, x_2), (y_1, y_2)]
        supports a third coordinate if it appears.'''
        assert (pt1.shape == pt2.shape)
        keys = ["x", "y"]
        if pt1.shape[0] == 3:  # append a third coordinate
            keys.append("z")
        coord_dict = {}
        for i in range(pt1.shape[0]):
            coord_dict[keys[i]] = np.array([pt1[i], pt2[i]])
        return coord_dict

    def plot_skeleton(self, frame_idx, data_source):
        for idx, n in enumerate(self.skeleton_dict["name"]):
            print(n, self.skeleton_dict["parent"][idx])
            if self.skeleton_dict["parent"][idx] is not None:  # if bodypart has a parent
                name = self.points[data_source][n][frame_idx, :]
                parent = self.points[data_source][self.skeleton_dict["parent"][idx]][frame_idx, :]
                coord_dict = self.pts_to_coords(name, parent)
                self.ax.plot(coord_dict["x"],
                             coord_dict["y"],
                             coord_dict["z"],
                             'gray',
                             marker=None)

    def vid_from_images(self, video_name=None):
        # TODO: some of these, like -crf 25, or frame rate, can be control parameters.
        # TODO: switch to ffmpeg-python?, see https://github.com/kkroening/ffmpeg-python/tree/master/examples
        string = " "
        cmd = string.join(["ffmpeg -r 2 -i",
                           os.path.join(str(self.image_folder),
                                        'im_%04d.{}'.format(self.image_type)),
                           "-vcodec libx264 -crf 25",
                           video_name])
        print(cmd)
        # cmd = "ffmpeg -r 2 -i images/IBL/im_%04d.jpg -vcodec libx264 -crf 25 ibl_pca3_hard_frames_maha_9.mp4"
        os.system(cmd)

    def clear_axes(self):
        for ax in self.ax:
            ax.cla()

    def compute_azim_elevs_traj(self):
        elevs = self.init_elev * np.sin(np.linspace(0., 1., self.num_frames) * (2 * np.pi) / 1.0)
        azims = self.init_azim + np.arange(self.num_frames)  # assuming that at 360 degs we rotate
        return elevs, azims

    def __call__(self, change_azim_elev=True, skeleton_data_sources=None, video_name=None):
        # TODO: decide how to save the figs. or just the ax. an Idea, have an upper level class that handels image saving etc.
        # TODO: if want to parallelize, then the azims should be precomputed.
        elevs, azims = self.compute_azim_elevs_traj()  # precomputed (lengthscale maybe slow for long vids.)
        for fr in tqdm(range(self.num_frames)):
            self.ax.clear()
            if change_azim_elev:
                self.set_curr_elev_azim(elev=elevs[fr], azim=azims[fr])
                # self.ax.azim += 1
            self.set_xyz_lims()
            self.set_labels()
            self.ax.set_title('Frame %i' % fr)
            if skeleton_data_sources is not None:
                for data_source in skeleton_data_sources:
                    self.plot_skeleton(fr, data_source)
            self.scatter_points(fr)

            self.fig.savefig(self.image_folder / ('im_%04d.%s' % (fr, self.image_type)))  # was png
            # save a png / jpg to folder given in self.__init__()
        if video_name is not None:
            self.vid_from_images(str(video_name))