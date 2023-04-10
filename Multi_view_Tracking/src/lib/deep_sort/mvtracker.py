from __future__ import absolute_import
from .tracker import Tracker
from deep_sort import nn_matching

class MVTracker:
    def __init__(self, view_ls, sv_threshold=0.3):
        self.mvtrack_dict = {}
        self.max_cosine_distance = sv_threshold
        self.nn_budget = None
        self.matching_mat = None
        self.next_id = [1]
        self.view_ls = view_ls
        for view in view_ls:
            self.mvtrack_dict[view] = Tracker(nn_matching.NearestNeighborDistanceMetric(
        "cosine", self.max_cosine_distance, self.nn_budget), max_iou_distance=0.7, next_id=self.next_id)

    def update(self, matching_mat):
        self.matching_mat = matching_mat