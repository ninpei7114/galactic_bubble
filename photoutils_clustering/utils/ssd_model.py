# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Scripts implementing SSD, DBox and MultiBoxLoss
This script for creating the SSD is created with reference to https://github.com/YutaroOgawa/pytorch_advanced
"""

from itertools import product as product
from math import sqrt as sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function

from utils.match import match


# Create a 34-layer vgg
def make_vgg():
    layers = []
    in_channels = 2  # Number of colour channels

    Conv2_1 = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Conv2_2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    # Conv2_2 = nn.Sequential(nn.BatchNorm2d(64), nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    Maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    Conv2_3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Conv2_4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     Conv2_4 = nn.Sequential(nn.BatchNorm2d(128),
    #               nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    Maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    Conv2_5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Conv2_6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     Conv2_6 = nn.Sequential(nn.BatchNorm2d(256),
    #               nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    Conv2_7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    Conv2_8 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Conv2_9 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     Conv2_9 = nn.Sequential(nn.BatchNorm2d(512),
    #               nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    Conv2_10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    Conv2_11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Conv2_12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     Conv2_12 = nn.Sequential(nn.BatchNorm2d(512),
    #               nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
    Conv2_13 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    Maxpool_5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
    Conv2_14 = nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
    Conv2_15 = nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
    #     Conv2_15 = nn.Sequential(nn.BatchNorm2d(1024),
    #               nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1)))

    layers += [
        Conv2_1,
        nn.ReLU(),
        Conv2_2,
        nn.ReLU(),
        Maxpool_1,
        Conv2_3,
        nn.ReLU(),
        Conv2_4,
        nn.ReLU(),
        Maxpool_2,
        Conv2_5,
        nn.ReLU(),
        Conv2_6,
        nn.ReLU(),
        Conv2_7,
        nn.ReLU(),
        Maxpool_3,
        Conv2_8,
        nn.ReLU(),
        Conv2_9,
        nn.ReLU(),
        Conv2_10,
        nn.ReLU(),
        Maxpool_4,
        Conv2_11,
        nn.ReLU(),
        Conv2_12,
        nn.ReLU(),
        Conv2_13,
        nn.ReLU(),
        Maxpool_5,
        Conv2_14,
        nn.ReLU(),
        Conv2_15,
        nn.ReLU(),
    ]

    return nn.ModuleList(layers)


# Create an extras module consisting of eight layers
def make_extras():
    layers = []
    in_channels = 1024  # Number of image channels input to extra output from the vgg module

    # Configuration for setting the number of channels in the convolution layer of the extra module
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]  # 0
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]  # #1
    #     layers += [nn.MaxPool2d(kernel_size=2)]  #Maxpoolは自分で付け足した   #2
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]  # 2
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]  #  #3
    #     layers += [nn.MaxPool2d(kernel_size=2)]  #Maxpoolは自分で付け足した #5
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]  # 4
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]  # 5
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]  # 6
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]  # 7

    return nn.ModuleList(layers)


# Create loc_layers to output the offset of the default box,
# and conf_layers to output the probability of each class for the default box
def make_loc_conf(num_classes=2, bbox_aspect_num=[4, 4, 4, 4, 4, 4]):
    loc_layers = []
    conf_layers = []

    # 22nd layer of VGG, convolution layer for conv4_3 (source1)
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]

    # Convolution layer for the final layer of VGG (source2)
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]

    # Convolution layer for source3 of extra
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]

    # Convolution layer for source4 of extra
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    # Convolution layer for source5 of extra
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    # Convolution layer for source6 of extra
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=1, padding=0)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


# Layer normalising the output from convC4_3 with L2Norm of scale=20
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  # Value to be set as the initial value for weight
        self.reset_parameters()  # Parameter initialisation
        self.eps = 1e-10

    def reset_parameters(self):
        """Execute initialisation to set the combined parameter to the value of the size scale"""
        init.constant_(self.weight, self.scale)

    def forward(self, x):
        """
        Calculate the root of the sum of squares over 512 channels for 38x38 features
        Layer to normalise each feature using 38x38 values and then multiply by the coefficients
        """

        # Calculate the sum of squares of the 38 x 38 features in each channel in the channel direction
        # Further route, divide and normalise them
        # The tensor size of norm is torch.Size([batch_num, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # Multiply by coefficients. One coefficient per channel, with 512 coefficients.
        # The tensor size of self.weight is torch.Size([512]), so deform it to torch.Size([batch_num, 512, 38, 38])
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


# Class to output the default box
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # Initial setting
        self.image_size = cfg["input_size"]  # 300 of image size
        # [38, 19, …] Size of the feature map for each source
        self.feature_maps = cfg["feature_maps"]
        self.num_priors = len(cfg["feature_maps"])  # Number of source = 6
        self.steps = cfg["steps"]  # [8, 16, …] pixel size of DBox
        self.min_sizes = cfg["min_sizes"]  # [30, 60, …] pixel size of small square DBox
        self.max_sizes = cfg["max_sizes"]  # [60, 111, …] pixel size of large square DBox
        self.aspect_ratios = cfg["aspect_ratios"]  # aspect ratio of the rectangular DBox.

    def make_dbox_list(self):
        """DBoxを作成する"""
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # Image size of the feature map
                # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k = self.image_size / self.steps[k]

                # Centre coordinates of the DBox (x,y), however normalised from 0 to 1.
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # Small DBox with aspect ratio 1 [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # Large DBox with aspect ratio 1 [cx,cy, width, height]
                # 'max_sizes': [45, 99, 153, 207, 261, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # Other aspect ratio DBoxes [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # Convert DBox to tensor [ torch.Size([8732, 4])]
        output = torch.Tensor(mean).view(-1, 4)

        # If the size of the DBox is greater than 1, set it to 1
        output.clamp_(max=1, min=0)

        return output


# Function to convert a DBox to a BBox using offset information
def decode(loc, dbox_list):
    """
    Parameters
    ----------
    loc:  [8732,4]
        Offset information to be inferred in the SSD model。
    dbox_list: [8732,4]
        DBoxの情報

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBoxの情報
    """

    # DBox is stored in [cx, cy, width, height]
    # loc is stored in [Δcx, Δcy, Δwidth, Δheight]

    # Calculate the BBox from the offset information
    boxes = torch.cat(
        (dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:], dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1
    )
    # The size of the boxes is torch.Size([8732, 4])

    # Converts the coordinate information of the BBox from [cx, cy, width, height] to [xmin, ymin, xmax, ymax
    boxes[:, :2] -= boxes[:, 2:] / 2  # Convert to coordinates (xmin,ymin)
    boxes[:, 2:] += boxes[:, :2]  # Convert to coordinates (xmax,ymax)

    return boxes


def decode_all(loc_data, dbox_list):
    """
    this function is used to determine the F1_score
    Unify the calculations for each threshold
    """
    boxes = torch.zeros_like(loc_data)
    for i, loc in enumerate(loc_data):
        boxes[i] = decode(loc, dbox_list)
    return boxes


# Functions for Non-Maximum Suppression
def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Removes BBoxes in 'boxes' that overlap too much (more than 'overlap').

    Parameters
    ----------
    boxes : [Number of BBoxes exceeding the confidence threshold (0.01), 4]
        BBox information.
    scores : [Number of BBoxes exceeding the confidence threshold (0.01)]
        Confidence information.

    Returns
    -------
    keep : list
        Indexes that passed the non-maximum suppression (nms) in descending order of confidence.
    count : int
        Number of BBoxes that passed the nms.
    """

    # Create a template for the return
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep: torch.Size([Number of BBoxes exceeding the confidence threshold]), all elements are 0

    # Calculate the area of each BBox
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # Sort scores in ascending order
    v, idx = scores.sort(0)

    # Extract the indexes of the top 'top_k' BBoxes (200 BBoxes) (there may be less than 200)
    idx = idx[-top_k:]
    # Loop as long as the number of elements in idx is not 0
    while idx.numel() > 0:
        i = idx[-1]  # The index with the current maximum confidence is 'i'

        # Store the index with the maximum confidence at the end of 'keep'
        # BBoxes that overlap with this BBox will be removed
        keep[count] = i
        count += 1

        # If it's the last BBox, break the loop
        if idx.size(0) == 1:
            break

        # Since the index with the current maximum confidence was stored in 'keep', reduce idx by one
        idx = idx[:-1]
        idx = update_index(area, boxes, idx, i, overlap)

    return keep, count


@torch.jit.script
def update_index(area, boxes, idx, i: int, overlap: float):
    minxy = boxes[i, 0:2].repeat(2)
    maxxy = boxes[i, 2:4].repeat(2)
    clamped = boxes[idx].clamp(min=minxy, max=maxxy)
    inter = (clamped[:, 2] - clamped[:, 0]) * (clamped[:, 3] - clamped[:, 1])

    # Calculation of IoU = intersect part / (area(a) + area(b) - intersect part)
    rem_areas = torch.index_select(area, 0, idx)  # Original area of each BBox
    union = (rem_areas - inter) + area[i]  # Area of the AND of the two areas
    IoU = inter / union

    # Only keep idx where IoU is less than overlap
    return idx[IoU.le(overlap)]  # 'le' performs the Less than or Equal to operation
    # dx where IoU is greater than overlap is eliminated
    # because it boxes the same object as the idx first selected and stored in keep


# Outputs BBoxes with overlaps removed from the conf and loc outputs during SSD inference
class Detect(Function):
    def __init__(self, conf_thresh=0.3, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)  # Prepared for normalizing conf with softmax function
        self.conf_thresh = conf_thresh  # Only handles DBoxes with conf higher than conf_thresh=0.01
        self.top_k = top_k  # In nm_supression, uses the top_k number with high conf, top_k = 200
        # In nm_supression, if IOU is larger than nms_thresh=0.45, it is considered as a BBox to the same object
        self.nms_thresh = nms_thresh

    def __call__(self, loc_data, conf_data, dbox_list, decoded=False, softmaxed=False):
        """
        Perform forward propagation calculation.

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            Offset information.
        conf_data: [batch_num, 8732,num_classes]
            Detection confidence.
        dbox_list: [8732,4]
            DDBox information.

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            (batch_num, class, top 200 of conf, BBox information)
        """

        num_batch = loc_data.size(0)  # Mini Batch size
        num_dbox = loc_data.size(1)  # Number of DBoxes = 8732
        num_classes = conf_data.size(2)  # Number of classes = 21

        # conf normalises after applying softmax
        if not softmaxed:
            conf_data = self.softmax(conf_data)

        # Create the type of output. Tensor size is [minibatch number, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # cof_dataを[batch_num,8732,num_classes]から[batch_num, num_classes,8732]に順番変更
        conf_preds = conf_data.transpose(2, 1)

        # Loop for each minibatch
        for i in range(num_batch):
            # 1. Calculate the corrected BBox [xmin, ymin, xmax, ymax] from loc and DBox
            if decoded:
                decoded_boxes = loc_data[i]
            else:
                decoded_boxes = decode(loc_data[i].to("cpu"), dbox_list.to("cpu"))

            # Create a copy of conf
            conf_scores = conf_preds[i].clone().to("cpu")

            # Loop for each image class (do not calculate for the background class index 0, start from index=1)
            for cl in range(1, num_classes):
                # 2. Extract BBox that exceeds the conf threshold
                # Create a mask of whether the conf exceeds the threshold,
                # and get the index of conf that exceeds the threshold as c_mask
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # 'gt' is Greater than. By 'gt', those that exceed the threshold become 1, and those below become 0
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                # 'scores' is torch.Size([Number of BBoxes exceeding the threshold])
                scores = conf_scores[cl][c_mask]

                # If there is no conf exceeding the threshold, i.e., when scores=[], do nothing
                if scores.nelement() == 0:  # nelementで要素数の合計を求める
                    continue

                # Change the size of 'c_mask' so that it can be applied to 'decoded_boxes'
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # Apply 'l_mask' to 'decoded_boxes'
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # 'decoded_boxes[l_mask]' becomes one-dimensional, so
                # reshape it with 'view' to the size of (Number of BBoxes exceeding the threshold, 4)

                # 3. Perform Non-Maximum Suppression and remove overlapping BBoxes
                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                # 'ids': Indexes that passed the Non-Maximum Suppression in descending order of conf are stored
                # 'count': Number of BBoxes that passed the Non-Maximum Suppression

                # Store the result of Non-Maximum Suppression in 'output'
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 2, 200, 5])


# Create the SSD class


class SSD(nn.Module):
    def __init__(self, cfg=None):
        super(SSD, self).__init__()
        if cfg is None:
            cfg = {
                "num_classes": 2,  # Total number of classes including the background class
                "input_size": 300,  # Input size of the image
                "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # Types of aspect ratios of the DBox to be output
                "feature_maps": [38, 19, 10, 5, 3, 1],  # Image size of each source
                "steps": [8, 16, 32, 64, 100, 300],  # Determine the size of the DBOX
                "min_sizes": [30, 60, 111, 162, 213, 264],  # Determine the size of the DBOX
                "max_sizes": [60, 111, 162, 213, 264, 315],  # Determine the size of the DBOX
                "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            }
        self.num_classes = cfg["num_classes"]  # Number of classes = 21

        # Create the SSD network
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # Create DBox
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

    def forward(self, x):
        sources = list()  # Store inputs to loc and conf from source1 to source6
        loc = list()  # Store outputs of loc
        conf = list()  # Store outputs of conf

        # Compute up to conv4_3 of vgg
        for k in range(23):
            x = self.vgg[k](x)

        # Input the output of conv4_3 to L2Norm, create source1, and add to sources
        source1 = self.L2Norm(x)
        sources.append(source1)

        # Compute vgg to the end, create source2, and add to sources
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # Compute conv and ReLU of extras
        # Add source3 to source6 to sources

        for k, v in enumerate(self.extras):
            x = F.relu(v(x))

            if k % 2 == 1:  # If conv→ReLU→cov→ReLU is done, add to source
                sources.append(x)

        # Apply corresponding convolution once to each of source1 to source6
        # Get elements of multiple lists in for loop with zip
        # Since there are source1 to source6, the loop will run 6 times
        for x, l, c in zip(sources, self.loc, self.conf):
            # Permute is used to change the order of elements
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # Execute convolution with l(x) and c(x)
            # The output size of l(x) and c(x) is [batch_num, 4*number of aspect ratios, height of featuremap, width of featuremap]
            # The number of aspect ratios varies by source, so to make it easier, change the order and adjust
            # Use permute to change the order of elements,
            # to [minibatch size, number of featuremaps, number of featuremaps, 4*number of aspect ratios]
            # (Note)
            # torch.contiguous() is a command to rearrange elements continuously in memory.
            # The view function will be used later.
            # To perform this view, the target variable needs to be continuously arranged in memory.

        # Further transform the shape of loc and conf
        # The size of loc becomes, torch.Size([batch_num, 34928])
        # The size of conf becomes torch.Size([batch_num, 183372])
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # Further adjust the shape of loc and conf
        # The size of loc becomes, torch.Size([batch_num, 8732, 4])
        # The size of conf becomes, torch.Size([batch_num, 8732, 2])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # Finally output
        output = (loc, conf, self.dbox_list)

        return output, torch.cat(
            [decode(loc[rr].to("cpu"), self.dbox_list)[None] for rr in range(loc.shape[0])], axis=0
        )
        # The return value is a tuple of (loc, conf, dbox_list)


class MultiBoxLoss(nn.Module):
    """This is the loss function class for SSD."""

    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device="cpu"):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  # 0.5, the threshold of the Jaccard coefficient for the match function
        self.negpos_ratio = neg_pos  # 3:1, the ratio of negative to positive for Hard Negative Mining
        self.device = device  # Whether to compute on CPU or GPU

    def forward(self, predictions, targets):
        """
        Calculation of the loss function.

        Parameters
        ----------
        predictions : Output of SSD net during training (tuple)
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 2]), dbox_list=torch.Size [8732,4])。

        targets : [num_batch, num_objs, 5]
            5 indicates the annotation information of the correct answer [xmin, ymin, xmax, ymax, label_ind]
        target is the correct label
        Returns
        -------
        loss_l : Tensor
            The value of the loss of loc
        loss_c : Tensor
            The value of the loss of conf

        """

        # The output of the SSD model is a tuple, so we unpack it
        loc_data, conf_data, dbox_list = predictions

        # Understanding the number of elements
        num_batch = loc_data.size(0)  # Size of the mini-batch
        num_dbox = loc_data.size(1)  # Number of DBoxes = 8732
        num_classes = conf_data.size(2)  # Number of classes = 2

        # conf_t_label: Store the label of the BBox closest to each DBox
        # loc_t: Store the location information of the BBox closest to each DBox
        conf_t_label = torch.zeros(num_batch, num_dbox).to(self.device, dtype=torch.long)
        loc_t = torch.zeros(num_batch, num_dbox, 4).to(self.device)
        # Prepare the default box as a new variable
        dbox = dbox_list.to(self.device)

        for idx in range(num_batch):  # ミニバッチでループ
            # Get the BBox and label of the correct annotation of the current mini-batch
            if len(targets[idx]) == 0:
                pass
            else:
                truths = targets[idx][:, :-1].to(self.device)  # BBox
                # Labels [label of object 1, label of object 2, ...]
                labels = targets[idx][:, -1].to(self.device)

                variance = [0.1, 0.2]
                # This variance is the coefficient used when correcting the calculation from DBox to BBox
                match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)

        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])

        # Reshape pos_mask to the size of loc_data
        # pos_mask : torch.size([num_batch, 8732]) → torch.size([num_batch, 8732, 1])
        # pos_idx : torch.size([num_batch, 8732, 4])
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # Get the loc_data and teacher data loc_t of the Positive DBox
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # Calculate the loss (error) of the offset information loc_t of the Positive DBox that found the object
        loss_l = torch.nan_to_num(F.smooth_l1_loss(loc_p, loc_t))
        ####  loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # ----------
        # Calculate the loss of class prediction: loss_c
        # Calculate the loss with the cross-entropy error function. However, since there are overwhelmingly many DBoxes where the background class is correct,
        # Perform Hard Negative Mining to make the ratio of the object discovery DBox to the background class DBox 1:3.
        # Therefore, among those predicted as the background class DBox, those with small losses are excluded from the class prediction loss.
        # ----------
        batch_conf = conf_data.view(-1, num_classes)

        # Calculate the loss function of class prediction (do not take the sum and do not crush the dimension by setting reduction='none')
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")

        # -----------------
        # We will create a mask to extract what we will extract from the Negative DBox with Hard Negative Mining
        # -----------------

        # Make the loss of the Positive DBox that found the object 0
        # (Note) The object has a label of 1 or more. Label 0 is the background.
        num_pos = pos_mask.long().sum(1, keepdim=True)  # Number of object class predictions per mini-batch
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0  # The DBox that found the object has a loss of 0
        # ↑ Extract the one with the largest loss among the Non-Ring

        # Implement Hard Negative Mining
        # Find idx_rank, which is the rank of the magnitude of loss_c for each DBox
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox, min=20)

        # idx_rank contains the rank of the magnitude of each DBox's loss from the top
        # Create a mask to take the DBox with a lower rank (i.e., larger loss) than the number of background DBoxes num_neg
        # torch.Size([num_batch, 8732])
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------
        # (End) From here, create a mask to extract the ones to be extracted by Hard Negative Mining from Negative DBox
        # -----------------

        # Shape the mask and match it to conf_data
        # pos_idx_mask is a mask to extract the conf of Positive DBox
        # neg_idx_mask is a mask to extract the conf of Negative DBox extracted by Hard Negative Mining
        # pos_mask: torch.Size([num_batch, 8732]) -> pos_idx_mask: torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # Extract only pos and neg from conf_data and make it conf_hnm. The shape is torch.Size([num_pos+num_neg, 21])
        # (Note) gt is an abbreviation for greater than (>). This extracts the index where the mask is 1.
        # pos_idx_mask+neg_idx_mask is addition, but it just summarizes the mask to the index.
        # In other words, whether it's pos or neg, make a list of those with a mask of 1 by addition, and get it with gt

        # Similarly, extract only pos and neg from the teacher data conf_t_label and make it conf_t_label_hnm
        # The shape becomes torch.Size([pos+neg])

        # Calculate the loss function of confidence (find the sum of elements = sum)

        loss_c_pos = torch.nan_to_num(
            F.cross_entropy(
                conf_data[(pos_idx_mask).gt(0)].view(-1, num_classes),
                ####                                conf_t_label[(pos_mask).gt(0)], reduction='sum'))
                conf_t_label[(pos_mask).gt(0)],
            )
        )
        loss_c_neg = torch.nan_to_num(
            F.cross_entropy(
                conf_data[(neg_idx_mask).gt(0)].view(-1, num_classes),
                ####                                conf_t_label[(neg_mask).gt(0)], reduction='sum'))
                conf_t_label[(neg_mask).gt(0)],
            )
        )
        loss_c = loss_c_pos + loss_c_neg

        # Divide the loss by the number N of BBoxes that found the object (total of all mini-batches)

        ##################
        # 参考にした本では、Nで割っているが、今回の学習ではNon-Ringのように、
        # positiveが0のデータもあるため、nanが出る可能性がある。
        #### N = num_pos.sum()
        #### loss_l /= N
        #### loss_c /= N
        ##################
        #         loss_c = 2*loss_c_pos/N + loss_c_neg/N
        return {
            "loc_loss": loss_l,
            "conf_loss": loss_c,
            "conf_loss_positive": loss_c_pos,
            "conf_loss_negative": loss_c_neg,
        }
