import os

import cv2
import skimage
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import measure
import cv2
import numpy as np
from scipy.ndimage import filters, measurements, distance_transform_edt
from scipy.ndimage.morphology import (binary_dilation, binary_fill_holes,
                                      distance_transform_cdt,
                                      distance_transform_edt)
from skimage.morphology import remove_small_objects, watershed

def compute_iou(prediction, ground_truth):
    smooth = 1e-5
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()
    intersection = (prediction * ground_truth).sum()
    return (intersection + smooth) / (prediction.sum() + ground_truth.sum() - intersection + smooth)


def accuracy(prediction, ground_truth):
    prediction = prediction.flatten()
    ground_truth = ground_truth.flatten()
    return (prediction == ground_truth).sum() / len(ground_truth)


def test_all_case(net, image_list, patch_size=(400, 400), stride_hw=(60, 60), save_result=True, test_save_path=None):
    total_metric = 0.0
    for image_path in tqdm(image_list):
        img = np.array(Image.open(image_path[0]))
        lab = np.array(Image.open(image_path[1]).convert("L"))
        lab[lab > 0] = 1
        prediction = test_single_case(net, img, patch_size, stride_hw)
        if np.sum(prediction) == 0:
            single_metric = (0, 0, 0, 0, 0, 0)
        else:
            single_metric = calculate_metric_percase(prediction, lab)
        total_metric += np.asarray(single_metric)
        if save_result:
            case_name = image_path[1].split(".")[0] + "_pred.png"
            prediction[prediction > 0.5] = 255
            cv2.imwrite(case_name, prediction)
    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))
    return avg_metric


def test_single_case(net, x, patch_size=(400, 400), stride_hw=(60, 60)):
    stridew, strideh = stride_hw
    winW, winH = patch_size
    shapeW, shapeH, channel = x.shape
    score_map = np.zeros((shapeW, shapeH))
    item_counting = np.zeros((shapeW, shapeH))
    for h in range((shapeH-winH) // strideh + 1):
        for w in range((shapeW-winW) // stridew + 1):
            test_patch = x[w*stridew:w*stridew+winW, h*strideh:h*strideh+winH]
            test_patch = (test_patch / test_patch.max())
            test_patch = test_patch.transpose((2, 0, 1))
            test_patch = torch.from_numpy(
                np.array([test_patch])).float().cuda()
            patch_output = net(test_patch)
            patch_output = patch_output.squeeze().squeeze().cpu().data.numpy()
            score_map[w*stridew:w*stridew+winW, h *
                      strideh:h*strideh+winH] += patch_output
            item_counting[w*stridew:w*stridew +
                          winW, h*strideh:h*strideh+winH] += 1
    score_map = score_map / item_counting
    score_map[score_map >= 0.5] = 1
    score_map[score_map < 0.5] = 0
    return score_map.astype(np.uint8)


def calculate_metric_percase(pred, gt):
    aji = fast_aji(gt, pred)
    iou = compute_iou(pred, gt)
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return aji, iou, dice, jc, hd, asd


def fast_aji(anno_arr, pred_arr):
    anno_arr = post_processing(anno_arr)
    anno_arr = measure.label(anno_arr)
    pred_arr = measure.label(pred_arr)
    gs, g_areas = np.unique(anno_arr, return_counts=True)
    assert np.all(gs == np.arange(len(gs)))
    ss, s_areas = np.unique(pred_arr, return_counts=True)
    assert np.all(ss == np.arange(len(ss)))

    i_idx, i_cnt = np.unique(np.concatenate([anno_arr.reshape(1, -1), pred_arr.reshape(1, -1)]),
                             return_counts=True, axis=1)
    i_arr = np.zeros(shape=(len(gs), len(ss)), dtype=np.int)
    i_arr[i_idx[0], i_idx[1]] += i_cnt
    u_arr = g_areas.reshape(-1, 1) + s_areas.reshape(1, -1) - i_arr
    iou_arr = 1.0 * i_arr / u_arr

    i_arr = i_arr[1:, 1:]
    u_arr = u_arr[1:, 1:]
    iou_arr = iou_arr[1:, 1:]

    j = np.argmax(iou_arr, axis=1)

    c = np.sum(i_arr[np.arange(len(gs) - 1), j])
    u = np.sum(u_arr[np.arange(len(gs) - 1), j])
    used = np.zeros(shape=(len(ss) - 1), dtype=np.int)
    used[j] = 1
    u += (np.sum(s_areas[1:] * (1 - used)))
    return 1.0 * c / u


def post_processing(pred):
    lab = pred
    distance = distance_transform_edt(pred)
    blb_raw = lab
    dst_raw = distance
    blb = np.copy(blb_raw)
    blb[blb >  0.5] = 1
    blb[blb <= 0.5] = 0
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1   

    dst_raw[dst_raw < 0] = 0
    dst = np.copy(dst_raw)
    dst = dst * blb
    dst[dst  > 0.5] = 1
    dst[dst <= 0.5] = 0

    marker = dst.copy()
    marker = binary_fill_holes(marker) 
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
    proced_pred = watershed(-dst_raw, marker, mask=blb)
    proced_pred[proced_pred > 0] = 1 
    return proced_pred


