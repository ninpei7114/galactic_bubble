# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Scripts implementing SSD, DBox and MultiBoxLoss
"""

from itertools import product as product
from math import sqrt as sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function

from utils.match import match


# 34層のvggモジュールを作成
def make_vgg():
    layers = []
    in_channels = 2  # 色チャネル数
    # なぜか自分で実装している
    Conv2_1 = nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
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


# 8層にわたる、extrasモジュールを作成
def make_extras():
    layers = []
    in_channels = 1024  # vggモジュールから出力された、extraに入力される画像チャネル数

    # extraモジュールの畳み込み層のチャネル数を設定するコンフィギュレーション
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


# デフォルトボックスのオフセットを出力するloc_layers、
# デフォルトボックスに対する各クラスの確率を出力するconf_layersを作成


def make_loc_conf(num_classes=2, bbox_aspect_num=[4, 4, 4, 4, 4, 4]):
    loc_layers = []
    conf_layers = []

    # VGGの22層目、conv4_3（source1）に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0] * num_classes, kernel_size=3, padding=1)]

    # VGGの最終層（source2）に対する畳み込み層
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1] * num_classes, kernel_size=3, padding=1)]

    # extraの（source3）に対する畳み込み層
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2] * num_classes, kernel_size=3, padding=1)]

    # extraの（source4）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3] * num_classes, kernel_size=3, padding=1)]

    # extraの（source5）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4] * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4] * num_classes, kernel_size=3, padding=1)]

    # extraの（source6）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5] * 4, kernel_size=1, padding=0)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5] * num_classes, kernel_size=1, padding=0)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


# convC4_3からの出力をscale=20のL2Normで正規化する層
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__()  # 親クラスのコンストラクタ実行
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale  # 係数weightの初期値として設定する値
        self.reset_parameters()  # パラメータの初期化
        self.eps = 1e-10

    def reset_parameters(self):
        """結合パラメータを大きさscaleの値にする初期化を実行"""
        init.constant_(self.weight, self.scale)  # weightの値がすべてscale（=20）になる

    def forward(self, x):
        """38x38の特徴量に対して、512チャネルにわたって2乗和のルートを求めた
        38x38個の値を使用し、各特徴量を正規化してから係数をかけ算する層"""

        # 各チャネルにおける38×38個の特徴量のチャネル方向の2乗和を計算し、
        # さらにルートを求め、割り算して正規化する
        # normのテンソルサイズはtorch.Size([batch_num, 1, 38, 38])になります
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # 係数をかける。係数はチャネルごとに1つで、512個の係数を持つ
        # self.weightのテンソルサイズはtorch.Size([512])なので
        # torch.Size([batch_num, 512, 38, 38])まで変形します
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


# デフォルトボックスを出力するクラス
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        # 初期設定
        self.image_size = cfg["input_size"]  # 画像サイズの300
        # [38, 19, …] 各sourceの特徴量マップのサイズ
        self.feature_maps = cfg["feature_maps"]
        self.num_priors = len(cfg["feature_maps"])  # sourceの個数=6
        self.steps = cfg["steps"]  # [8, 16, …] DBoxのピクセルサイズ
        self.min_sizes = cfg["min_sizes"]  # [30, 60, …] 小さい正方形のDBoxのピクセルサイズ
        self.max_sizes = cfg["max_sizes"]  # [60, 111, …] 大きい正方形のDBoxのピクセルサイズ
        self.aspect_ratios = cfg["aspect_ratios"]  # 長方形のDBoxのアスペクト比

    def make_dbox_list(self):
        """DBoxを作成する"""
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # fまでの数で2ペアの組み合わせを作る　f_P_2 個
                # 特徴量の画像サイズ
                # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k = self.image_size / self.steps[k]

                # DBoxの中心座標 x,y　ただし、0～1で規格化している
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # アスペクト比1の小さいDBox [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # アスペクト比1の大きいDBox [cx,cy, width, height]
                # 'max_sizes': [45, 99, 153, 207, 261, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # その他のアスペクト比のdefBox [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]

        # DBoxをテンソルに変換 torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        # DBoxの大きさが1を超えている場合は1にする
        output.clamp_(max=1, min=0)

        return output


# オフセット情報を使い、DBoxをBBoxに変換する関数
def decode(loc, dbox_list):
    """
    オフセット情報を使い、DBoxをBBoxに変換する。

    Parameters
    ----------
    loc:  [8732,4]
        SSDモデルで推論するオフセット情報。
    dbox_list: [8732,4]
        DBoxの情報

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBoxの情報
    """

    # DBoxは[cx, cy, width, height]で格納されている
    # locも[Δcx, Δcy, Δwidth, Δheight]で格納されている

    # オフセット情報からBBoxを求める
    boxes = torch.cat(
        (dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:], dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1
    )
    # boxesのサイズはtorch.Size([8732, 4])となります

    # BBoxの座標情報を[cx, cy, width, height]から[xmin, ymin, xmax, ymax] に
    boxes[:, :2] -= boxes[:, 2:] / 2  # 座標(xmin,ymin)へ変換
    boxes[:, 2:] += boxes[:, :2]  # 座標(xmax,ymax)へ変換

    return boxes


# Non-Maximum Suppressionを行う関数


def decode_all(loc_data, dbox_list):
    """
    F1 scoreを求める時に使用する。
    Thresholdごとの計算を一本化する。
    sub.py/calc_f1scoreで使用
    """
    boxes = torch.zeros_like(loc_data)
    for i, loc in enumerate(loc_data):
        boxes[i] = decode(loc, dbox_list)
    return boxes


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    Non-Maximum Suppressionを行う関数。
    boxesのうち被り過ぎ（overlap以上）のBBoxを削除する。

    Parameters
    ----------
    boxes : [確信度閾値（0.01）を超えたBBox数,4]
        BBox情報。
    scores :[確信度閾値（0.01）を超えたBBox数]
        confの情報

    Returns
    -------
    keep : リスト
        confの降順にnmsを通過したindexが格納
    count : int
        nmsを通過したBBoxの数
    """

    # returnのひな形を作成
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    # keep：torch.Size([確信度閾値を超えたBBox数])、要素は全部0

    # 各BBoxの面積areaを計算
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # socreを昇順に並び変える
    v, idx = scores.sort(0)

    # 上位top_k個（200個）のBBoxのindexを取り出す（200個存在しない場合もある）
    idx = idx[-top_k:]
    # idxの要素数が0でない限りループする
    while idx.numel() > 0:
        i = idx[-1]  # 現在のconf最大のindexをiに

        # keepの現在の最後にconf最大のindexを格納する
        # このindexのBBoxと被りが大きいBBoxをこれから消去する
        keep[count] = i
        count += 1

        # 最後のBBoxになった場合は、ループを抜ける
        if idx.size(0) == 1:
            break

        # 現在のconf最大のindexをkeepに格納したので、idxをひとつ減らす
        idx = idx[:-1]
        idx = update_index(area, boxes, idx, i, overlap)

    return keep, count


@torch.jit.script
def update_index(area, boxes, idx, i: int, overlap: float):
    minxy = boxes[i, 0:2].repeat(2)
    maxxy = boxes[i, 2:4].repeat(2)
    clamped = boxes[idx].clamp(min=minxy, max=maxxy)
    inter = (clamped[:, 2] - clamped[:, 0]) * (clamped[:, 3] - clamped[:, 1])

    # IoU = intersect部分 / (area(a) + area(b) - intersect部分)の計算
    rem_areas = torch.index_select(area, 0, idx)  # 各BBoxの元の面積
    union = (rem_areas - inter) + area[i]  # 2つのエリアのANDの面積
    IoU = inter / union

    # IoUがoverlapより小さいidxのみを残す
    return idx[IoU.le(overlap)]  # leはLess than or Equal toの処理をする演算です
    # IoUがoverlapより大きいidxは、最初に選んでkeepに格納したidxと同じ物体に対してBBoxを囲んでいるため消去


# SSDの推論時にconfとlocの出力から、被りを除去したBBoxを出力する


class Detect(Function):
    def __init__(self, conf_thresh=0.3, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)  # confをソフトマックス関数で正規化するために用意
        self.conf_thresh = conf_thresh  # confがconf_thresh=0.01より高いDBoxのみを扱う
        self.top_k = top_k  # nm_supressionでconfの高いtop_k個を計算に使用する, top_k = 200
        self.nms_thresh = nms_thresh  # nm_supressionでIOUがnms_thresh=0.45より大きいと、同一物体へのBBoxとみなす

    def __call__(self, loc_data, conf_data, dbox_list, decoded=False, softmaxed=False):
        """
        順伝搬の計算を実行する。

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            オフセット情報。
        conf_data: [batch_num, 8732,num_classes]
            検出の確信度。
        dbox_list: [8732,4]
            DBoxの情報

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            （batch_num、クラス、confのtop200、BBoxの情報）
        """

        num_batch = loc_data.size(0)  # ミニバッチのサイズ
        num_dbox = loc_data.size(1)  # DBoxの数 = 8732
        num_classes = conf_data.size(2)  # クラス数 = 21

        # confはソフトマックスを適用して正規化する
        # confはソフトマックスを適用して正規化する
        if not softmaxed:
            conf_data = self.softmax(conf_data)

        # 出力の型を作成する。テンソルサイズは[minibatch数, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # cof_dataを[batch_num,8732,num_classes]から[batch_num, num_classes,8732]に順番変更
        conf_preds = conf_data.transpose(2, 1)

        # ミニバッチごとのループ
        for i in range(num_batch):
            # 1. locとDBoxから修正したBBox [xmin, ymin, xmax, ymax] を求める
            if decoded:
                decoded_boxes = loc_data[i]
            else:
                decoded_boxes = decode(loc_data[i].to("cpu"), dbox_list.to("cpu"))

            # confのコピーを作成
            conf_scores = conf_preds[i].clone().to("cpu")

            # 画像クラスごとのループ（背景クラスのindexである0は計算せず、index=1から）
            for cl in range(1, num_classes):
                # 2.confの閾値を超えたBBoxを取り出す
                # confの閾値を超えているかのマスクを作成し、
                # 閾値を超えたconfのインデックスをc_maskとして取得
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                # gtはGreater thanのこと。gtにより閾値を超えたものが1に、以下が0になる
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                # scoresはtorch.Size([閾値を超えたBBox数])
                scores = conf_scores[cl][c_mask]

                # 閾値を超えたconfがない場合、つまりscores=[]のときは、何もしない
                if scores.nelement() == 0:  # nelementで要素数の合計を求める
                    continue

                # c_maskを、decoded_boxesに適用できるようにサイズを変更します
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                # l_maskをdecoded_boxesに適応します
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # decoded_boxes[l_mask]で1次元になってしまうので、
                # viewで（閾値を超えたBBox数, 4）サイズに変形しなおす

                # 3. Non-Maximum Suppressionを実施し、被っているBBoxを取り除く
                ids, count = nm_suppression(boxes, scores, self.nms_thresh, self.top_k)
                # ids：confの降順にNon-Maximum Suppressionを通過したindexが格納
                # count：Non-Maximum Suppressionを通過したBBoxの数

                # outputにNon-Maximum Suppressionを抜けた結果を格納
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 2, 200, 5])


# SSDクラスを作成する


class SSD(nn.Module):
    def __init__(self, cfg=None):
        super(SSD, self).__init__()
        if cfg is None:
            cfg = {
                "num_classes": 2,  # 背景クラスを含めた合計クラス数
                "input_size": 300,  # 画像の入力サイズ
                "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
                "feature_maps": [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
                "steps": [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
                "min_sizes": [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
                "max_sizes": [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
                "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
            }
        self.num_classes = cfg["num_classes"]  # クラス数=21

        # SSDのネットワークを作る
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

    def forward(self, x):
        sources = list()  # locとconfへの入力source1～6を格納
        loc = list()  # locの出力を格納
        conf = list()  # confの出力を格納

        # vggのconv4_3まで計算する
        for k in range(23):
            x = self.vgg[k](x)

        # conv4_3の出力をL2Normに入力し、source1を作成、sourcesに追加
        source1 = self.L2Norm(x)
        sources.append(source1)

        # vggを最後まで計算し、source2を作成、sourcesに追加
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # extrasのconvとReLUを計算
        # source3～6を、sourcesに追加

        for k, v in enumerate(self.extras):
            x = F.relu(v(x))

            if k % 2 == 1:  # conv→ReLU→cov→ReLUをしたらsourceに入れる
                sources.append(x)
        #

        # source1～6に、それぞれ対応する畳み込みを1回ずつ適用する
        # zipでforループの複数のリストの要素を取得
        # source1～6まであるので、6回ループが回る
        #         print(self.loc, self.conf)
        for x, l, c in zip(sources, self.loc, self.conf):
            # Permuteは要素の順番を入れ替え
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            # l(x)とc(x)で畳み込みを実行
            # l(x)とc(x)の出力サイズは[batch_num, 4*アスペクト比の種類数, featuremapの高さ, featuremap幅]
            # sourceによって、アスペクト比の種類数が異なり、面倒なので順番入れ替えて整える
            # permuteで要素の順番を入れ替え、
            # [minibatch数, featuremap数, featuremap数,4*アスペクト比の種類数]へ
            # （注釈）
            # torch.contiguous()はメモリ上で要素を連続的に配置し直す命令です。
            # あとでview関数を使用します。
            # このviewを行うためには、対象の変数がメモリ上で連続配置されている必要があります。

        # さらにlocとconfの形を変形
        # locのサイズは、torch.Size([batch_num, 34928])
        # confのサイズはtorch.Size([batch_num, 183372])になる
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # さらにlocとconfの形を整える
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 2])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        #         print(conf[:,:,1])
        # 最後に出力する
        output = (loc, conf, self.dbox_list)

        return output, torch.cat(
            [decode(loc[rr].to("cpu"), self.dbox_list)[None] for rr in range(loc.shape[0])], axis=0
        )
        # 返り値は(loc, conf, dbox_list)のタプル


class MultiBoxLoss(nn.Module):
    """SSDの損失関数のクラスです。"""

    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device="cpu"):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh  # 0.5 関数matchのjaccard係数の閾値
        self.negpos_ratio = neg_pos  # 3:1 Hard Negative Miningの負と正の比率
        self.device = device  # CPUとGPUのいずれで計算するのか

    def forward(self, predictions, targets):
        """
        損失関数の計算。

        Parameters
        ----------
        predictions : SSD netの訓練時の出力(tuple)
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 2]), dbox_list=torch.Size [8732,4])。

        targets : [num_batch, num_objs, 5]
            5は正解のアノテーション情報[xmin, ymin, xmax, ymax, label_ind]を示す
        targetは、正解ラベル
        Returns
        -------
        loss_l : テンソル
            locの損失の値
        loss_c : テンソル
            confの損失の値

        """

        # SSDモデルの出力がタプルになっているので、個々にばらす
        loc_data, conf_data, dbox_list = predictions

        # 要素数を把握
        num_batch = loc_data.size(0)  # ミニバッチのサイズ
        num_dbox = loc_data.size(1)  # DBoxの数 = 8732
        num_classes = conf_data.size(2)  # クラス数 = 2

        # conf_t_label：各DBoxに一番近い正解のBBoxのラベルを格納させる
        # loc_t:各DBoxに一番近い正解のBBoxの位置情報を格納させる
        conf_t_label = torch.zeros(num_batch, num_dbox).to(self.device, dtype=torch.long)
        loc_t = torch.zeros(num_batch, num_dbox, 4).to(self.device)
        # デフォルトボックスを新たな変数で用意
        dbox = dbox_list.to(self.device)

        for idx in range(num_batch):  # ミニバッチでループ
            # 現在のミニバッチの正解アノテーションのBBoxとラベルを取得
            if len(targets[idx]) == 0:
                pass
            else:
                truths = targets[idx][:, :-1].to(self.device)  # BBox
                # ラベル [物体1のラベル, 物体2のラベル, …]
                labels = targets[idx][:, -1].to(self.device)

                variance = [0.1, 0.2]
                # このvarianceはDBoxからBBoxに補正計算する際に使用する式の係数です
                match(self.jaccard_thresh, truths, dbox, variance, labels, loc_t, conf_t_label, idx)

        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])

        # pos_maskをloc_dataのサイズに変形
        # pos_mask : torch.size([num_batch, 8732]) → torch.size([num_batch, 8732, 1])
        # pos_idx : torch.size([num_batch, 8732, 4])
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # Positive DBoxのloc_dataと、教師データloc_tを取得
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 物体を発見したPositive DBoxのオフセット情報loc_tの損失（誤差）を計算
        loss_l = torch.nan_to_num(F.smooth_l1_loss(loc_p, loc_t))
        ####  loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # ----------
        # クラス予測の損失：loss_cを計算
        # 交差エントロピー誤差関数で損失を計算する。ただし、背景クラスが正解であるDBoxが圧倒的に多いので、
        # Hard Negative Miningを実施し、物体発見DBoxと背景クラスDBoxの比が1:3になるようにする。
        # そこで背景クラスDBoxと予想したもののうち、損失が小さいものは、クラス予測の損失から除く
        # ----------
        batch_conf = conf_data.view(-1, num_classes)

        # クラス予測の損失を関数を計算(reduction='none'にして、和をとらず、次元をつぶさない)
        loss_c = F.cross_entropy(batch_conf, conf_t_label.view(-1), reduction="none")

        # -----------------
        # これからNegative DBoxのうち、Hard Negative Miningで抽出するものを求めるマスクを作成します
        # -----------------

        # 物体発見したPositive DBoxの損失を0にする
        # （注意）物体はlabelが1以上になっている。ラベル0は背景。
        num_pos = pos_mask.long().sum(1, keepdim=True)  # ミニバッチごとの物体クラス予測の数
        loss_c = loss_c.view(num_batch, -1)  # torch.Size([num_batch, 8732])
        loss_c[pos_mask] = 0  # 物体を発見したDBoxは損失0とする
        # ↑ Non-Ringのうち、最もlossが大きいものを抽出していく

        # Hard Negative Miningを実施する
        # 各DBoxの損失の大きさloss_cの順位であるidx_rankを求める
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox, min=20)

        # idx_rankは各DBoxの損失の大きさが上から何番目なのかが入っている
        # 背景のDBoxの数num_negよりも、順位が低い（すなわち損失が大きい）DBoxを取るマスク作成
        # torch.Size([num_batch, 8732])
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------
        # （終了）これからNegative DBoxのうち、Hard Negative Miningで抽出するものを求めるマスクを作成します
        # -----------------

        # マスクの形を整形し、conf_dataに合わせる
        # pos_idx_maskはPositive DBoxのconfを取り出すマスクです
        # neg_idx_maskはHard Negative Miningで抽出したNegative DBoxのconfを取り出すマスクです
        # pos_mask：torch.Size([num_batch, 8732])→pos_idx_mask：torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_dataからposとnegだけを取り出してconf_hnmにする。形はtorch.Size([num_pos+num_neg, 21])
        #### conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)
        ####                       ].view(-1, num_classes)
        # （注釈）gtは greater than (>)の略称。これでmaskが1のindexを取り出す。
        # pos_idx_mask+neg_idx_maskは足し算だが、indexへのmaskをまとめているだけである。
        # つまり、posであろうがnegであろうが、マスクが1のものを足し算で一つのリストにし、それをgtで取得

        # 同様に教師データであるconf_t_labelからposとnegだけを取り出してconf_t_label_hnmに
        # 形はtorch.Size([pos+neg])になる
        #### conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]
        #### loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

        # confidenceの損失関数を計算（要素の合計=sumを求める）

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

        # 物体を発見したBBoxの数N（全ミニバッチの合計）で損失を割り算

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
