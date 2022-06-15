from __future__ import absolute_import
from .tracker import Tracker
from deep_sort import nn_matching

class MVTracker:
    def __init__(self, view_ls):
        self.mvtrack_dict = {}
        self.max_cosine_distance = 0.2
        self.nn_budget = None
        self.matching_mat = None
        self.next_id = [1]
        self.view_ls = view_ls
        for view in view_ls:
            self.mvtrack_dict[view] = Tracker(nn_matching.NearestNeighborDistanceMetric(
        "cosine", self.max_cosine_distance, self.nn_budget), next_id=self.next_id)

    def update(self, matching_mat):
        self.matching_mat = matching_mat