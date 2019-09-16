import numpy as np
import utils.evaluation
import matplotlib.pyplot as plt

def remove(pred_label, cb):
    # how to use filter to constrain not removing big nucleus
    out = pred_label.copy()
    out_f = out.ravel()  # flatten out
    component_sizes = np.bincount(out_f)
    cb_ = cb
#     cb_ = []
#     for c in cb:
#         if component_sizes[c] < 400:  # 357 this should be larger???
#             cb_.append(c)
            
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            for c in cb_:
                if out[i, j] == c:
                    out[i, j] = 0
    
    return out

def remove_border(pred_label):
    # find the classes of the partial nuclei which located on the border
    # traverse the border
    cb = set()
    for i in range(512):
        if pred_label[i, 0] != 0:
            cb.add(pred_label[i, 0])
        if pred_label[0, i] != 0:
            cb.add(pred_label[0, i])
        if pred_label[i, 511] != 0:
            cb.add(pred_label[i, 511])
        if pred_label[511, i] != 0:
            cb.add(pred_label[511, i])
    
    # change these classes to 0 as background
    # maybe I should add a constrain about not filtering the big cell
    return remove(pred_label, cb)

def show(ground_truth, prediction, threshold=0.5, image_name="N"):
    # Compute Intersection over Union
    IOU = utils.evaluation.intersection_over_union(ground_truth, prediction)

    # Create diff map
    diff = np.zeros(ground_truth.shape + (3,))  # become 3 channels
    diff2 = np.zeros(ground_truth.shape + (3,))  # become 3 channels
    
    A = ground_truth.copy()
    B = prediction.copy()
    A[A > 0] = 1
    B[B > 0] = 1
#     D = A - B
#     diff2[D > 0, 0] = 1
#     diff2[D < 0, 2] = 1
    
    # Object-level errors
    C = IOU.copy()
    C[C >= threshold] = 1
    C[C < threshold] = 0
    missed = np.where(np.sum(C, axis=1) == 0)[0]  
    extra = np.where(np.sum(C, axis=0) == 0)[0]

    for m in missed:
        diff[ground_truth == m+1, 0] = 1
    for e in extra:
        diff[prediction == e+1, 2] = 1
        
    matches = IOU > 0.1  # why here only use 0.1, 0.4 or 0.5 seems more reasonable
    merges = np.where(np.sum(matches, axis=0) > 1)[0]  # pred label <-> 2 gt
    splits = np.where(np.sum(matches, axis=1) > 1)[0]  # gt <-> 2 pred
    
    for m in merges:
        diff2[prediction == m+1, 0] = 1
    for s in splits:
        diff2[ground_truth == s+1, 2] = 1
    
    # Display figures
    fig, ax = plt.subplots(1, 5, figsize=(20,4))
    ax[0].imshow(ground_truth)
    ax[0].set_title("Ground Truth objects: " + str(len(np.unique(ground_truth))))
    ax[1].imshow(diff)
    ax[1].set_title("Segment errors: miss {}, extra {}.".format(str(len(missed)), str(len(extra))))
    ax[2].imshow(prediction)
    ax[2].set_title("Predict objects:" + str(len(np.unique(prediction))))
    ax[3].imshow(IOU)
    ax[3].set_title(image_name + " IOU")
    ax[4].imshow(diff2)
    ax[4].set_title("Segment errors: merges {}, splits {}.".format(str(len(merges)), str(len(splits))))
