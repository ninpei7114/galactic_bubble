from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from utils.ssd_model import Detect, decode_all


class EarlyStopping_f1_score:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path, flog, patience=10, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.f1_score_max = -np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.flog = flog

    def __call__(self, f1_score, model):
        score = f1_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.flog.write(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(f1_score, model)
            self.counter = 0

    def save_checkpoint(self, f1_score, model):
        """Saves model when f1_score increase."""
        if self.verbose:
            self.trace_func(f"f1_score increase ({self.f1_score_max:.6f} --> {f1_score:.6f}).  Saving model ...")
            self.flog.write(f"f1_score increase ({self.f1_score_max:.6f} --> {f1_score:.6f}).  Saving model ...\n")
        torch.save(model.state_dict(), self.path)
        self.f1_score_max = f1_score


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path, flog, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.flog = flog

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.flog.write(f"EarlyStopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
            self.flog.write(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n"
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


# ll , boxは一枚の画像に対する、正解と予想
def calc_collision(ll, box, iou=0.5):
    """
    1. この関数は、SSDが予想したBBoxと正解labelのBBoxの重なり率を計算する。
        ll: 正解ラベル  [xmin, ymin, xmax, ymax]。
            複数ラベルの場合は、[[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ----]

        box : モデルの予想結果、[[conf, xmin, ymin, xmax, ymax], [conf, xmin, ymin, xmax, ymax], -----]の配列

    2. またNon-Ring領域を正しく判断できているかの計算も行う。
        0: 正解でも予測でもない
        1: 正解だが予測ではない
        2: 正解ではないが予測されている
        3: 正解かつ予測
    """
    true_positive = []

    # Ringのみの結果を取り出す
    box = box[1, :, :].detach().numpy()

    # 各boxの面積を求める。
    area = (box[:, 3] - box[:, 1]) * (box[:, 4] - box[:, 2])

    ### Non-RingのF1 scoreを計算する ###
    # x = np.zeros((300,300), np.uint8)
    # for t_box in box:
    #     x[int(t_box[2]*300):int(t_box[4]*300), int(t_box[1]*300):int(t_box[3]*300)] |= 2

    for l in ll:
        ## lのxminなどの順番は上記を参照
        # x[int(l[3]*300):int(l[1]*300), int(l[2]*300):int(l[0]*300)] |= 1
        # 正解boxの面積
        l_area = (l[2] - l[0]) * (l[3] - l[1])

        # 重なり部分の面積を求める
        abx_mn = np.maximum(l[0], box[:, 1])  # xmin
        aby_mn = np.maximum(l[1], box[:, 2])  # ymin
        abx_mx = np.minimum(l[2], box[:, 3])  # xmax
        aby_mx = np.minimum(l[3], box[:, 4])  # ymax

        w = np.maximum(0, abx_mx - abx_mn)
        h = np.maximum(0, aby_mx - aby_mn)

        intersect = w * h
        IoU = intersect / (area + l_area - intersect)

        # 重なりが0.45以上のbox
        true_positive.append(IoU > iou)

    # Non_Ring_TN = (x==0).sum()
    # Non_Ring_FN = (x==1).sum()
    # Non_Ring_FP = (x==2).sum()
    # Non_Ring_TP = (x==3).sum()

    # Non_Ring_precision = Non_Ring_TP / (Non_Ring_TP + Non_Ring_FP)
    # Non_Ring_recall = Non_Ring_TP / (Non_Ring_TP + Non_Ring_FN)

    # Non_Ring_judge = [Non_Ring_precision, Non_Ring_recall]

    # Non_Ring_judge = [Non_Ring_TP, Non_Ring_FP, Non_Ring_FN, Non_Ring_TN]

    if len(ll) == 0:
        return 0, box[:, 0], False  # box[:,0]は、probability
    else:
        return np.stack(true_positive), box[:, 0], True  # box[:,0]は、probability


def calc_f1score(val_seikai, val_bbbb, mode, jaccard=0.45, top_k=50, iou=0.5):
    """
    TP1=推定したボックスのうち、正解と一定以上のIoUを持つ個数
    FP=推定したボックスのうち、正解と一定以上のIoUを持たない個数
    TP2=正解のうち、一定以上のIoUを推定したボックスと持つ個数
    FN=正解のうち、一定以上のIoUを推定したボックスと持たない個数

    TP1_FP : モデルのboxの数（conf閾値適応済み）, predict showした時の赤枠

    """
    if mode == "train":
        thresholds = [i / 20 for i in range(6, 16, 1)]
    else:
        thresholds = [i / 20 for i in range(6, 16, 1)]

    f1_score = -10000
    f1_score_non_ring = -10000
    threthre = 0
    threthre_noring = 0
    PRE = []
    RE = []
    TP1_l = []
    FP_l = []

    dbox_list = val_bbbb[2].cpu()
    val_bbbb = [decode_all(val_bbbb[0].cpu(), dbox_list), nn.Softmax(dim=-1)(val_bbbb[1].cpu()), dbox_list]

    for th in thresholds:
        TP1 = 0
        TP2 = 0
        TP1_FP = 0
        TP1_FP_non_ring = 0
        TP2_FN = 0

        # TP_NonRing = 0
        # FP_NonRing = 0
        # FN_NonRing = 0

        # detectクラスで、nm_suppressionする
        detect = Detect(conf_thresh=th, nms_thresh=jaccard, top_k=top_k)
        output = detect(*val_bbbb, decoded=True, softmaxed=True)
        collisions = [calc_collision(s, b, iou=iou) for s, b in zip(val_seikai, output)]
        # colは面積のあたり判定
        for col, prob, flag in collisions:
            ## このflagは Ring か NonRingの画像かの判断をしている
            if flag:
                idx = prob > th
                ## 正解boxとDBoxで、重なりがiou以上でかつ、probが閾値を超えるもの

                tp1 = col.any(axis=0)
                tp2 = col.any(axis=1)  # for tp2

                tp1_fp = idx.sum()
                tp2_fn = col.shape[0]

                TP1 += np.sum(tp1)
                TP2 += np.sum(tp2)

                TP1_FP += tp1_fp
                TP1_FP_non_ring += tp1_fp
                TP2_FN += tp2_fn

                # TP_NonRing += non_ring_judge[0]
                # FP_NonRing += non_ring_judge[1]
                # FN_NonRing += non_ring_judge[2]

            else:
                idx = prob > th
                TP1_FP_non_ring += idx.sum()

        PRE.append(TP1 / TP1_FP)
        RE.append(TP2 / TP2_FN)
        TP1_l.append(TP1)
        FP_l.append(TP1_FP - TP1)

        f1_score_ = calc_f1_sub(
            TP1, TP2, TP1_FP, TP2_FN
        )  # + calc_f1_sub(TP_NonRing, TP_NonRing, TP_NonRing+FP_NonRing, TP_NonRing+FN_NonRing)
        f1_score_non_ring_ = calc_f1_sub(TP1, TP2, TP1_FP_non_ring, TP2_FN)

        if f1_score_ > f1_score:
            f1_score = f1_score_
            threthre = th

        if f1_score_non_ring_ > f1_score_non_ring:
            f1_score_non_ring = f1_score_non_ring_
            threthre_noring = th

    return f1_score, threthre, f1_score_non_ring, threthre_noring  # , PRE, RE, TP1_l, FP_l


def calc_f1_sub(TP1, TP2, TP1_FP, TP2_FN):
    r1 = TP1 / (TP1_FP + 1e-9)
    r2 = TP2 / (TP2_FN + 1e-9)
    # print(f'precision : {r1}, recall : {r2}')
    return 2 * r1 * r2 / (r1 + r2 + 1e-9)


def print_and_log(f, moji):
    if isinstance(moji, list):
        for i in moji:
            print(i)
            f.write(i + "\n")
    else:
        print(moji)
        f.write(moji + "\n")


def transfer_resnet(net, param_path):
    deepcluster_weight = torch.load(param_path)
    net.vgg[0].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.0.weight"])
    net.vgg[0].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.0.bias"])

    # net.vgg[2][0].weight = nn.Parameter(deepcluster_weight['state_dict']['features.2.0.weight'])
    # net.vgg[2][0].bias = nn.Parameter(deepcluster_weight['state_dict']['features.2.0.bias'])
    # net.vgg[2][0].running_mean = deepcluster_weight['state_dict']['features.2.0.running_mean'].to('cpu')
    # net.vgg[2][0].running_var = deepcluster_weight['state_dict']['features.2.0.running_var'].to('cpu')
    # net.vgg[2][0].num_batches_tracked = deepcluster_weight['state_dict']['features.2.0.num_batches_tracked'].to('cpu')
    # net.vgg[2][1].weight = nn.Parameter(deepcluster_weight['state_dict']['features.2.1.weight'])
    # net.vgg[2][1].bias = nn.Parameter(deepcluster_weight['state_dict']['features.2.1.bias'])
    net.vgg[2].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.2.weight"])
    net.vgg[2].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.2.bias"])

    net.vgg[5].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.5.weight"])
    net.vgg[5].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.5.bias"])

    net.vgg[7][0].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.7.0.weight"])
    net.vgg[7][0].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.7.0.bias"])
    net.vgg[7][0].running_mean = deepcluster_weight["state_dict"]["features.7.0.running_mean"].to("cpu")
    net.vgg[7][0].running_var = deepcluster_weight["state_dict"]["features.7.0.running_var"].to("cpu")
    net.vgg[7][0].num_batches_tracked = deepcluster_weight["state_dict"]["features.7.0.num_batches_tracked"].to("cpu")
    net.vgg[7][1].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.7.1.weight"])
    net.vgg[7][1].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.7.1.bias"])
    # net.vgg[7].weight = nn.Parameter(deepcluster_weight['state_dict']['features.7.1.weight'])
    # net.vgg[7].bias = nn.Parameter(deepcluster_weight['state_dict']['features.7.1.bias'])

    net.vgg[10].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.10.weight"])
    net.vgg[10].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.10.bias"])

    net.vgg[12][0].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.12.0.weight"])
    net.vgg[12][0].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.12.0.bias"])
    net.vgg[12][0].running_mean = deepcluster_weight["state_dict"]["features.12.0.running_mean"].to("cpu")
    net.vgg[12][0].running_var = deepcluster_weight["state_dict"]["features.12.0.running_var"].to("cpu")
    net.vgg[12][0].num_batches_tracked = deepcluster_weight["state_dict"]["features.12.0.num_batches_tracked"].to(
        "cpu"
    )
    net.vgg[12][1].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.12.1.weight"])
    net.vgg[12][1].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.12.1.bias"])
    # net.vgg[12].weight = nn.Parameter(deepcluster_weight['state_dict']['features.12.1.weight'])
    # net.vgg[12].bias = nn.Parameter(deepcluster_weight['state_dict']['features.12.1.bias'])

    net.vgg[17].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.17.weight"])
    net.vgg[17].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.17.bias"])

    net.vgg[19][0].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.19.0.weight"])
    net.vgg[19][0].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.19.0.bias"])
    net.vgg[19][0].running_mean = deepcluster_weight["state_dict"]["features.19.0.running_mean"].to("cpu")
    net.vgg[19][0].running_var = deepcluster_weight["state_dict"]["features.19.0.running_var"].to("cpu")
    net.vgg[19][0].num_batches_tracked = deepcluster_weight["state_dict"]["features.19.0.num_batches_tracked"].to(
        "cpu"
    )
    net.vgg[19][1].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.19.1.weight"])
    net.vgg[19][1].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.19.1.bias"])
    # net.vgg[19].weight = nn.Parameter(deepcluster_weight['state_dict']['features.19.1.weight'])
    # net.vgg[19].bias = nn.Parameter(deepcluster_weight['state_dict']['features.19.1.bias'])

    net.vgg[21].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.21.weight"])
    net.vgg[21].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.21.bias"])

    net.vgg[24].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.24.weight"])
    net.vgg[24].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.24.bias"])

    net.vgg[26][0].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.26.0.weight"])
    net.vgg[26][0].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.26.0.bias"])
    net.vgg[26][0].running_mean = deepcluster_weight["state_dict"]["features.26.0.running_mean"].to("cpu")
    net.vgg[26][0].running_var = deepcluster_weight["state_dict"]["features.26.0.running_var"].to("cpu")
    net.vgg[26][0].num_batches_tracked = deepcluster_weight["state_dict"]["features.26.0.num_batches_tracked"].to(
        "cpu"
    )
    net.vgg[26][1].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.26.1.weight"])
    net.vgg[26][1].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.26.1.bias"])
    # net.vgg[26].weight = nn.Parameter(deepcluster_weight['state_dict']['features.26.1.weight'])
    # net.vgg[26].bias = nn.Parameter(deepcluster_weight['state_dict']['features.26.1.bias'])

    net.vgg[28].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.28.weight"])
    net.vgg[28].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.28.bias"])

    net.vgg[31].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.31.weight"])
    net.vgg[31].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.31.bias"])

    net.vgg[33].weight = nn.Parameter(deepcluster_weight["state_dict"]["features.33.weight"])
    net.vgg[33].bias = nn.Parameter(deepcluster_weight["state_dict"]["features.33.bias"])
