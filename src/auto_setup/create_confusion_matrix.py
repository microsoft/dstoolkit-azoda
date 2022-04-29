from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from util import adjust_text_location
from util import get_bbs
from util import get_best_overlap
from util import get_lastest_iteration
from util import plot_confusion_matrix
import argparse
import cv2
import numpy as np
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', help='Name of dataset/project')
args = parser.parse_args()
dataset = args.dataset

save_imgs = False
project_base = f'../../{args.dataset}/'
latest_pred = get_lastest_iteration(os.path.join(project_base, 'test_inferences'), req_prefix='labels_')
latest_gt = get_lastest_iteration(os.path.join(project_base, 'datasets'), req_prefix='test_')
df_pred = pd.read_csv(latest_pred)
df_gt = pd.read_csv(latest_gt)

# Iterate over images
images_directory = os.path.join(project_base, 'test_images')
visualisations_directory = os.path.join(project_base, 'visualisations')
confusion_matrix_directory = os.path.join(project_base, 'confusion_matrices')
os.makedirs(visualisations_directory, exist_ok=True)
os.makedirs(confusion_matrix_directory, exist_ok=True)
os.makedirs(os.path.join(visualisations_directory, 'output/'), exist_ok=True)
os.makedirs(os.path.join(visualisations_directory, 'output_fp/'), exist_ok=True)
os.makedirs(os.path.join(visualisations_directory, 'output_fn/'), exist_ok=True)

iou_thres = 0.5
actual = []
predicted = []
counter = 0
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 0.5
lineType = 2
plt.figure(figsize=(80, 16))
defect_names = list(set(df_gt['class']))

for filename in os.listdir(images_directory):
    counter += 1
    if filename.endswith(".jpg"):
        img = cv2.imread(os.path.join(images_directory, filename))
        for bb in get_bbs(df_gt, filename, defect_names):
            cv2.rectangle(img, (bb[0], bb[2]), (bb[1], bb[3]), (0, 255, 0), 2)
            cv2.putText(img, defect_names[bb[4]-1],
                        adjust_text_location((bb[0], bb[2]), img),
                        font,
                        fontScale,
                        (0, 255, 0),
                        lineType)

        for bb in get_bbs(df_pred, filename, defect_names):
            cv2.rectangle(img, (bb[0], bb[2]), (bb[1], bb[3]), (0, 0, 255), 2)
            cv2.putText(img, defect_names[bb[4]-1],
                        adjust_text_location((bb[1], bb[3]), img),
                        font,
                        fontScale,
                        (0, 0, 255),
                        lineType)
        if save_imgs:
            cv2.imwrite(f'{visualisations_directory}/output/{filename}', img)
        bbs_pred = get_bbs(df_pred, filename, defect_names)
        bbs_gt = get_bbs(df_gt, filename, defect_names)
        for bb in bbs_gt:
            iou, id, max_class = get_best_overlap(bb, bbs_pred,
                                                  iou_thres=iou_thres)
            actual.append(bb[4])
            predicted.append(max_class)
            if save_imgs:
                if not bb[4] == max_class:
                    cv2.imwrite(f'{visualisations_directory}/output_fn/{filename}', img)
            if max_class > 0:
                bbs_pred.pop(id)
        incorrect_count = 0
        for bb in bbs_pred:
            incorrect_count += 1
            actual.append(0)
            predicted.append(bb[4])
        if save_imgs:
            if incorrect_count > 0:
                cv2.imwrite(f'{visualisations_directory}/output_fp/{filename}', img)

# Display performance metrics
matrix = classification_report(actual,
                               predicted,
                               labels=[i+1 for i in range(len(defect_names))],
                               target_names=defect_names)
print('Classification report : \n', matrix)
time_stamp = datetime.now().strftime('%y%m%d%H%m%S')
cnf_matrix = confusion_matrix(actual,
                              predicted,
                              labels=[i for i in range(1+len(defect_names))])
np.set_printoptions(precision=2)
plt.figure(figsize=(16, 16))
export_path = f'{confusion_matrix_directory}/test_confusion_matrix_{time_stamp}.png'
plot_confusion_matrix(cnf_matrix,
                      classes=['Background'] + defect_names,
                      title='Test Confusion Matrix',
                      normalize=False,
                      export_path=export_path)
