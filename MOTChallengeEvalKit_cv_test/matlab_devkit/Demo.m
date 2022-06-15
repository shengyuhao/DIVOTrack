clear all
clc

[sequenceName, mets, metsID, additionalInfo, results]=evaluateTracking('D:\Code\CrossViewTracking\eval_result\yolo_det_motion\seqs.txt', 'D:\Code\CrossViewTracking\eval_result\yolo_det_motion\track\Auditorium_view-HC3.txt', 'D:\Code\CrossViewTracking\eval_result\yolo_det_motion\gt\Auditorium_view-HC3.txt', 'D:\Code\CrossViewTracking\eval_result\yolo_det_motion\gt\Auditorium_view-HC3', 'MOT17');