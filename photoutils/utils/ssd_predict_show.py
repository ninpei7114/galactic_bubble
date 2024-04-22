# -*- coding: utf-8 -*-
"""
第2章SSDで予測結果を画像として描画するクラス
普通のssd_predict_showとは違う
変更点
color_mean    (0,0,0) に変更

def ssd_predict　も変更

cv2.imreadを、画像を読み込むのではなく、np.loadに変更している
さらに、numpyに変更する時も変更している
modelに入力するデータに　　.float() を付け加えた
rgb_imageも変更した


"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.ssd_model import Detect


class SSDPredictShow:
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  # クラス名
        self.net = net  # SSDネットワーク
        self.detect = Detect()

        color_mean = (0, 0)  # (BGR)の色の平均値
        input_size = 300  # 画像のinputサイズを300×300にする  # 前処理クラス

    def show(self, image_file_path, data_confidence_level):
        """
        物体検出の予測結果を表示をする関数。

        Parameters
        ----------
        image_file_path:  str
            画像のファイルパス
        data_confidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(image_file_path, data_confidence_level)
        #         print(predict_bbox)

        fig = self.vis_bbox(
            rgb_img,
            bbox=predict_bbox,
            label_index=pre_dict_label_index,
            scores=scores,
            label_names=self.eval_categories,
        )

        return fig

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        SSDで予測させる関数。
        想定するデータのshapeは、(width, height, channel)

        Parameters
        ----------
        image_file_path:  strt
            画像のファイルパス

        dataconfidence_level: float
            予測で発見とする確信度の閾値

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # rgbの画像データを取得
        #         img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        #         height, width, channels = img.shape  # 画像のサイズを取得
        #         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = np.load(image_file_path)
        a, b, c = img.shape
        if a == 2 or a == 3:
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)
            width, height, channel = img.shape
        #             print(width, height, channel)
        else:
            width, height, channel = img.shape

        tempo_img = copy.deepcopy(img)
        if channel == 3:
            img = img[:, :, :2]

        ll = []
        for i in range(2):
            ds = tempo_img[:, :, i].copy()
            img_i_min = ds.min().copy()
            ds = ds - img_i_min
            img_i_max = ds.max().copy()
            img_i = (ds / img_i_max) * 300
            img_i[img_i > 255] = 255
            ll.append(img_i)

        rgb_img = np.uint8(
            np.concatenate([ll[0][:, :, None], ll[1][:, :, None], np.zeros([height, width, 1])], axis=2)
        )
        #         rgb_img = np.uint8(np.concatenate([ll[0][:,:,None], ll[1][:,:,None], ll[2][:,:,None]], axis=2))

        # 画像の前処理
        img = torch.from_numpy(img).permute(2, 0, 1)

        # SSDで予測
        self.net.eval()  # ネットワークを推論モードへ
        x = img.unsqueeze(0).float()  # ミニバッチ化：torch.Size([1, 3, 300, 300])

        with torch.no_grad():
            output, decoded_box = self.net(x)
            detections = self.detect(
                output[0], output[1], output[2]
            )  # , decode(loc[0], self.dbox_list), self.softmax(conf)

        # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値
        #         print(detections.shape)
        for i in range(1, 2):
            for k in range(0, 20):
                print(detections[0, i][k])

        # confidence_levelが基準以上を取り出す
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        # 条件以上の値を抽出
        #         print(detections[:, 0:, :, 0])
        #         print(detections[:, 0:, :, 0].shape)
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        #         print('find_index:' + str(find_index))
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 抽出した物体数分ループを回す
            if (find_index[1][i]) > 0:  # 背景クラスでないもの
                sc = detections[i][0]  # 確信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_indexはミニバッチ数、クラス、topのtuple
                lable_ind = find_index[1][i] - 1
                # （注釈）
                # 背景クラスが0なので1を引く

                # 返り値のリストに追加
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)
        print("scores:" + str(scores))
        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        物体検出の予測結果を画像で表示させる関数。

        Parameters
        ----------
        rgb_img:rgbの画像
            対象の画像データ
        bbox: list
            物体のBBoxのリスト
        label_index: list
            物体のラベルへのインデックス
        scores: list
            物体の確信度。
        label_names: list
            ラベル名の配列

        Returns
        -------
        なし。rgb_imgに物体検出結果が加わった画像が表示される。
        """

        # 枠の色の設定
        #         label_names.append('back')
        #         label_index.append(1)
        num_classes = len(label_names)  # クラス数（背景のぞく）
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        fig = plt.figure(figsize=(10, 10))

        #         currentAxis = plt.gca()
        currentAxis = fig.add_subplot(111)
        currentAxis.imshow(rgb_img)
        # BBox分のループ
        for i, bb in enumerate(bbox):
            # ラベル名
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

            # 枠につけるラベル　例：person;0.72
            if scores is not None:
                sc = scores[i]
                display_txt = "%s: %.2f" % (label_name, sc)
            else:
                display_txt = "%s: ans" % (label_name)

            # 枠の座標
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 長方形を描画する
            currentAxis.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 長方形の枠の左上にラベルを描画する
            currentAxis.text(xy[0], xy[1], display_txt, bbox={"facecolor": color, "alpha": 0.5})

        return fig
