import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def get_lastest_iteration(base_dir, req_prefix=''):
    filenames = [f for f in os.listdir(base_dir)
                 if os.path.isfile(os.path.join(base_dir, f))
                 and f.startswith(req_prefix)]
    print('filenames', filenames)
    return os.path.join(base_dir, sorted(filenames)[-1])


# Text placement helper function
def adjust_text_location(coords, img):
    buffer = 10
    height = img.shape[0]
    width = img.shape[1]
    coords = list(coords)
    if coords[0] < buffer:
        coords[0] = buffer
    if coords[1] < buffer:
        coords[1] = buffer
    if coords[1] > height-buffer:
        coords[1] = height-buffer
    if coords[0] > width-4*buffer:
        coords[0] = width-4*buffer
    return tuple(coords)


def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def get_bbs(df, filename, defect_names=['']):
    bbs = []
    filename_col = 'filename'
    defect_name_col = 'class'
    for _, row in df[df[filename_col] == filename].iterrows():
        class_id = row[defect_name_col]
        defect_names_count = len(defect_names)
        if class_id not in [i+1 for i in range(defect_names_count)]:
            if class_id in defect_names:
                class_id = defect_names.index(class_id)+1
            else:
                print('Unexpected class encountered:', class_id)
                class_id = -1
        if 'XMin' in row:
            bbs.append([row['XMin'], row['XMax'], row['YMin'], row['YMax'],
                        class_id])
        else:
            bbs.append([row['xmin'], row['xmax'], row['ymin'], row['ymax'],
                        class_id])
    return bbs


def get_iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    x2 = min(bb1[1], bb2[1])
    y1 = max(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    if x2 < x1 or y2 < y1:
        return 0

    intersection = (x2-x1)*(y2-y1)

    area1 = max(1, (bb1[1]-bb1[0])*(bb1[3]-bb1[2]))
    area2 = max(1, (bb2[1]-bb2[0])*(bb2[3]-bb2[2]))

    iou = intersection/(area1+area2-intersection)
    assert iou >= 0
    assert iou <= 1
    return iou


def get_best_overlap(bb_target, bb_list, iou_thres=0.5):
    max_iou = iou_thres
    max_id = -1
    max_class = 0
    for bb_id, bb in enumerate(bb_list):
        iou = get_iou(bb, bb_target)
        if iou > max_iou:
            max_class = bb[4]
            max_iou = get_iou(bb, bb_target)
            max_id = bb_id
    return max_iou, max_id, max_class


# from https://datascience.stackexchange.com/questions/40067/confusion-matrix-three-classes-python/40068
def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          export_path=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'  # if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if export_path == '':
        print('ERROR: Please specify export path')
    else:
        plt.savefig(export_path)
