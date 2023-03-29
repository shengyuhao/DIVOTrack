from deep_sort.detection import Detection
from sklearn import preprocessing as sklearn_preprocessing
from application_util import preprocessing
from sklearn.utils.extmath import softmax
from . import linear_assignment
from application_util import visualization
#from sklearn.utils.linear_assignment_ import linear_assignment as sklearn_linear_assignment
from scipy.optimize import linear_sum_assignment as sklearn_linear_assignment
import cv2
import numpy as np
import config as C 


class Update():
    def __init__(self, seq, mvtracker, display, view_list):
        self.seq = seq
        self.view_ls = mvtracker.view_ls
        self.tracker = mvtracker
        self.display = display
        self.min_confidence = 0.7
        self.nms_max_overlap = 1
        self.min_detection_height = 0
        self.delta = 0.5
        self.epsilon = 0.5
        self.result = {key: [] for key in self.view_ls}
        self.view_list = view_list
        matrix = [[]]



    def create_detections(self, detection_mat, frame_idx, min_height=0):
        if len(detection_mat) == 0:
            return []
        frame_indices = detection_mat[:, 0].astype(np.int)
        mask = frame_indices == frame_idx


        detection_list = []
        for row in detection_mat[mask]:
            bbox, confidence, feature, id = row[2:6], row[6], row[10:], row[1]
            if bbox[3] < min_height:
                continue
            detection_list.append(Detection(bbox, confidence, feature, id))
        return detection_list

    def select_detection(self, frame_idx, view):
        detections = self.create_detections(
            self.seq[view]["detections"], frame_idx, self.min_detection_height)   

        detections = [d for d in detections]# if d.confidence >= self.min_confidence]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        return detections

    def select_view_detection(self, frame_idx, view):
        view_detections = self.create_detections(
            self.seq[view]["view_detections"], frame_idx, self.min_detection_height)   

        view_detections = [d for d in view_detections]# if d.confidence >= self.min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in view_detections])
        scores = np.array([d.confidence for d in view_detections])
        indices = preprocessing.non_max_suppression(
            boxes, self.nms_max_overlap, scores)
        view_detections = [view_detections[i] for i in indices]
        return view_detections
        
    def frame_matching(self, frame_idx):
        def gen_X(features):
            features = [sklearn_preprocessing.normalize(i, axis=1) for i in features]
            all_blocks_X = {view: [] for view in self.view_ls}
            for x, view_x in zip(features, self.view_ls):
                row_blocks_X = {view: [] for view in self.view_ls}
                for y, view_y in zip(features, self.view_ls):
                    S12 = np.dot(x, y.transpose(1, 0))
                    scale12 = np.log(self.delta / (1 - self.delta) * S12.shape[1]) / self.epsilon
                    S12 = softmax(S12 * scale12)
                    S12[S12 < 0.5] = 0
                    assign_ls = sklearn_linear_assignment(- S12)
                    assign_ls = np.asarray(assign_ls)
                    assign_ls = np.transpose(assign_ls)
                    X_12 = np.zeros((S12.shape[0], S12.shape[1]))
                    for assign in assign_ls:
                        if S12[assign[0], assign[1]] != 0:
                            X_12[assign[0], assign[1]] = 1
                    row_blocks_X[view_y] = X_12
                all_blocks_X[view_x] = row_blocks_X
            
            return all_blocks_X
        # print("Matching frame %05d" % frame_idx)
        all_view_features = []
        all_view_id = []
        for view in self.view_ls:
            view_feature = []
            view_id = []
            self.tracker.mvtrack_dict[view].detections = self.select_detection(frame_idx, view)
            self.tracker.mvtrack_dict[view].view_detections = self.select_view_detection(frame_idx, view)
            for detection in self.tracker.mvtrack_dict[view].view_detections:
                view_feature.append(detection.feature)
                view_id.append(detection.id)
            if view_feature != []:
                view_feature = np.stack(view_feature)
                view_id = np.stack(view_id)
                view_feature = sklearn_preprocessing.normalize(view_feature, norm='l2', axis=1)
                all_view_features.append(view_feature)
            else:
                print(1)
                all_view_features.append(np.array([[0] * 512]))
            all_view_id.append(view_id)
        match_mat = gen_X(all_view_features)
        self.tracker.update(match_mat)

    def frame_callback(self, frame_idx):
        if C.RENEW_TIME:
            re_matching = frame_idx % C.RENEW_TIME == 0
        else:
            re_matching = 0
        for view in self.view_ls:
            self.tracker.mvtrack_dict[view].predict()
            if view == self.view_list[0]:
                self.tracker.mvtrack_dict[view].pre_update(False)
            else:
                self.tracker.mvtrack_dict[view].pre_update(re_matching)
        
        for view in self.view_ls:
            linear_assignment.spatial_association(self.tracker, view)
        track_ls = []
        for view in self.view_ls:
            track_ls += self.tracker.mvtrack_dict[view].matches
            track_ls += self.tracker.mvtrack_dict[view].possible_matches
            track_ls += self.tracker.mvtrack_dict[view].matches_backup
        track_ls = [i[0] for i in track_ls]
        for view in self. view_ls:
            for track_ in track_ls:
                if track_ in self.tracker.mvtrack_dict[view].unmatched_tracks:
                    self.tracker.mvtrack_dict[view].unmatched_tracks.remove(track_)
        for view in self.view_ls:
            self.tracker.mvtrack_dict[view].update()

    def frame_display(self, vis, frame_idx, view):

        # Update visualization.
        if self.display:
            image = cv2.imread(
                self.seq[view]["image_filenames"][frame_idx - self.seq[view]["min_frame_idx"]], cv2.IMREAD_COLOR)
            vis.set_image(image.copy(), view, str(frame_idx))
            # vis.draw_detections(detections)
            vis.draw_trackers(self.tracker.mvtrack_dict[view].tracks)

        # Store results.
        for track in self.tracker.mvtrack_dict[view].tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            self.result[view].append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    def run(self):
        if self.display:
            visualizer = visualization.Visualization(self.seq, update_ms=5)
        else:
            visualizer = visualization.NoVisualization(self.seq)
        print('start inference...')
        visualizer.run(self.frame_matching, self.frame_callback, self.frame_display)