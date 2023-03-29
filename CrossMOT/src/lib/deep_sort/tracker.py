# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=80, n_init=3, next_id=[1]):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.next_id = next_id

        self.matches = []
        self.matches_backup = []
        self.unmatched_tracks = []
        self.unmatched_detections = []
        self.possible_matches = []

        self.detections = None
        # self.view_detections = None

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def pre_update(self, re_matching):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.

        (
            self.matches,
            self.unmatched_tracks,
            self.unmatched_detections,
            self.matches_backup,
        ) = self._match(re_matching)
        self.matches = self.to_abs_idx(self.matches)
        self.matches_backup = self.to_abs_idx(self.matches_backup)
        self.unmatched_tracks = [self.tracks[i].track_id for i in self.unmatched_tracks]
        # return matches, unmatched_tracks, unmatched_detections

    def to_abs_idx(self, idx_pairs):
        abs_idx_pairs = []
        for pair in idx_pairs:
            abs_idx_pairs.append((self.tracks[pair[0]].track_id, pair[1]))
        del_ls = []
        for i, pair_i in enumerate(abs_idx_pairs):
            for j, pair_j in enumerate(abs_idx_pairs):
                if i < j:
                    if pair_i[0] == pair_j[0]:
                        del_ls.append(pair_j)
        ret = [i for i in abs_idx_pairs if i not in del_ls]
        return ret

    def update(self):
        matching2tracking = 1
        # Update track set.
        for track_idx, detection_idx in self.matches:
            for track in self.tracks:
                if track.track_id == track_idx:
                    track.update(self.kf, self.detections[detection_idx])
        for track_idx in self.unmatched_tracks:
            for track in self.tracks:
                if track.track_id == track_idx:
                    track.mark_missed()

        for bmatch in self.matches_backup[::-1]:
            for match in self.possible_matches:
                if bmatch[0] == match[0] or bmatch[1] == match[1]:
                    self.matches_backup.remove(bmatch)
                    break
            for detection_id in self.unmatched_detections:
                if detection_id == bmatch[1]:
                    self.unmatched_detections.remove(detection_id)

        self.possible_matches += self.matches_backup

        for detection_idx in self.unmatched_detections:
            track_idx = self._initiate_track(self.detections[detection_idx])
            self.matches.append((track_idx, detection_idx))

        for pmatch in self.possible_matches[::-1]:
            for match in self.matches:
                if pmatch[0] == match[0]:
                    self.possible_matches.remove(pmatch)
                    break

        for track_idx, association_idx in self.possible_matches:
            state = 1
            for track in self.tracks:
                if track.track_id == track_idx:
                    track.update(self.kf, self.detections[association_idx])
                    state = 0
            if state:
                self._associate_track(self.detections[association_idx], track_idx)
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, re_matching):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices
            )

            return cost_matrix

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]
        (
            matches_a,
            unmatched_tracks_a,
            unmatched_detections,
        ) = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            self.detections,
            confirmed_tracks,
        )
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        (
            matches_b,
            unmatched_tracks_b,
            unmatched_detections,
        ) = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            self.detections,
            iou_track_candidates,
            unmatched_detections,
        )
        matches = matches_a + matches_b
        unmatched_tracks_c = []
        unmatched_detections_b = []
        matches_backup = []
        if re_matching:
            unmatched_tracks_c = [i[0] for i in matches]
            unmatched_detections_b = [i[1] for i in matches]
            matches_backup = matches[:]
            matches = []
        unmatched_tracks = list(
            set(unmatched_tracks_a + unmatched_tracks_b + unmatched_tracks_c)
        )
        unmatched_detections += unmatched_detections_b
        return matches, unmatched_tracks, unmatched_detections, matches_backup

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                self.next_id[0],
                self.n_init,
                self.max_age,
                detection.feature,
            )
        )
        idx = self.next_id[0]
        self.next_id[0] += 1
        return idx

    def _associate_track(self, detection, track_idx):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(
                mean,
                covariance,
                track_idx,
                self.n_init,
                self.max_age,
                detection.feature,
            )
        )
