# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer
from tqdm import tqdm


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.view_ls = list(seq_info.keys())
        key0 = self.view_ls[0]
        self.frame_idx = seq_info[key0]["min_frame_idx"]
        self.last_idx = seq_info[key0]["max_frame_idx"]
        self.len = len(self.view_ls)
        self.view_id = 0

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_matching, frame_callback, frame_display):
        for frame_i in tqdm(range(self.frame_idx, self.last_idx)):
            frame_matching(frame_i)
            frame_callback(frame_i)
            for view_i in self.view_ls:
                frame_display(None, frame_i, view_i)

class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        self.view_ls = list(seq_info.keys())
        key0 = self.view_ls[0]
        image_shape = seq_info[key0]["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, 'visual')
        self.viewer.thickness = 2
        self.frame_idx = seq_info[key0]["min_frame_idx"]
        self.last_idx = seq_info[key0]["max_frame_idx"]
        self.len = len(self.view_ls)
        self.view_id = 0

    def run(self, frame_matching, frame_callback, frame_display):
        self.viewer.run(lambda: self._update_fun(frame_matching, frame_callback, frame_display))

    def _update_fun(self, frame_matching, frame_callback, frame_display):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        if self.view_id == 0:
            frame_matching(self.frame_idx)
            frame_callback(self.frame_idx)
        frame_display(self, self.frame_idx, self.view_ls[self.view_id])

        if self.view_id < self.len:
            self.view_id += 1
        if self.view_id == self.len:
            self.view_id = 0
            self.frame_idx += 1
        return True


    def set_image(self, image, view, frame_id):
        self.viewer.image = image
        self.viewer.view = view
        self.viewer.frame_id = frame_id

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#
