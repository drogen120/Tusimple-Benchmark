import json 
import numpy as np
import cv2
import matplotlib.pyplot as plt

TRAIN_PATH='./train_set/'
LABEL_NAME='label_data_0531.json'
json_gt = [json.loads(line) for line in open(TRAIN_PATH+LABEL_NAME).readlines()]
for gt in json_gt:
    #gt = json_gt[0]
    gt_lanes = gt['lanes']
    print len(gt_lanes)
    y_samples = gt['h_samples']
    print len(y_samples)
    raw_file = gt['raw_file']

    print raw_file
    print gt_lanes
    img = plt.imread(TRAIN_PATH+raw_file)
    plt.imshow(img)
    plt.show()

    gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane
                   in gt_lanes]

    img_vis = img.copy()

    print gt_lanes_vis

    for lane in gt_lanes_vis:
        for pt in lane:
            cv2.circle(img_vis, pt, radius=5, color=(0, 255, 0))

    plt.imshow(img_vis)
    plt.show()
