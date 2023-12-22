# -*- coding: utf-8 -*-
"""
https://github.com/amdegroot/ssd.pytorch
のbox_utils.pyより使用
関数matchを行うファイル

本章の実装はGitHub: amdegroot/ssd.pytorch [4] を参考にしています。
MIT License
Copyright (c) 2017 Max deGroot, Ellis Brown

"""


import torch


def point_form(boxes):
    """Convert prior_boxes to (xmin, ymin, xmax, ymax)
    jaccard係数を計算するために行う

    DBoxは作成時、(cx, cy, width, height)となっている。
    これを、(cx, cy, width, height) -> (xmin, ymin, xmax, ymax)に変換している。
    DBoxのshapeは、torch.Size([8732, 4])
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), 1)  # xmin, ymin  # xmax, ymax


def center_size(boxes):
    """Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2], 1)  # cx, cy  # w, h


def intersect(box_a, box_b):
    """We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
        box_a: (tensor) bounding boxes, Shape: [A,4].
        box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
        (tensor) intersection area, Shape: [A,B].
    重なり部分の計算
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    引数:
        box_a : 正解BBox(正解座標)
        box_b : (xmin, ymin, xmax, ymax)の並びになったDBox

    正解BBoxとDBoxのjaccard係数の計算方法例:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)

    jaccard係数を計算する。
    このjaccard係数を使用して、Positive DBoxとNegative DBoxに分ける。
    対象は、正解BBoxとDBox
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
    引数:
        truths → 正解BBox(正解座標), shapeは[num_batch, num_objs, 4]
        priors → DBox
        variances → DBoxからBBoxに補正計算する際に使用する係数
        labels → 正解BBoxのラベル(正解ラベル), shapeは[num_batch, num_objs, 1]
        loc_t → 0のテンソル、shapeは(num_batch, num_dbox, 4)
        conf_t → 0のテンソル、shapeは(num_batch, num_dbox)

    SSDの損失関数を定義する際に、
    まず8732個のDBoxから学習データの画像の正解BBoxと近いDBox（正解と物体クラスが一致かつ、座標情報も近いDBox）
    を抽出する。
    """
    # 正解BBoxと近いジャッカード係数を計算
    # point_form(priors)は、x_min, y_min, x_max, y_max
    overlaps = jaccard(truths, point_form(priors))
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)  # axis1方向の最大値と、そのindex
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)  # axis0方向の最大値と、そのindex
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]  # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    # jaccard係数が0.5以上のとなる正解BBoxを持たないDBoxは、Negative DBoxとする。
    # つまり、labelを0として背景クラスとして扱う。
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc  # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """
    DBoxからSSDの出力の(Δcx, Δcy, Δw, Δh)を使って、BBoxを作成する
    引数:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes

    返り値:
        encoded boxes (tensor), Shape: [num_priors, 4]

    cx = cx_d + 0.1* Δcx * w_d
    cy = cy_d + 0.1* Δcy * h_d
    w = w_d * exp(0.2 * Δw)
    h = h_d * exp(0.2 * Δh)

    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= variances[0] * priors[:, 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
