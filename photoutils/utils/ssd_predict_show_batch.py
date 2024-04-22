# -*- coding: utf-8 -*-
"""
第2章SSDで予測結果を画像として描画するクラス

cv2.imreadを、画像を読み込むのではなく、np.loadに変更している
さらに、numpyに変更する時も変更している
modelに入力するデータに .float() を付け加えた

"""
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.ssd_model import Detect


class SSDPredictShow:
    """SSDでの予測と画像の表示をまとめて行うクラス"""

    def __init__(self, net):
        self.eval_categories = ["Ring"]  # クラス名
        self.net = net  # SSDネットワーク
        self.detect = Detect()

    def show(self, img, data_confidence_level, save=False, img_file_path=None):
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
        batch, a, _, _ = img.shape

        if a == 2 or a == 3:
            img = np.swapaxes(img, 1, 3)
            img = np.swapaxes(img, 1, 2)
            batch, width, height, channel = img.shape
        else:
            batch, width, height, channel = img.shape
        if channel == 3:
            img = img[:, :, :, :2]
        predict_bbox_list = []
        for bat in range(batch):
            img_ = copy.deepcopy(img[bat])
            predict_bbox, pre_dict_label_index, scores = self.ssd_predict(img_, width, height, data_confidence_level)
            rgb_img = np.uint8(np.concatenate([img_, np.zeros([300, 300, 1])], axis=2) * 255)

            tempo_l = []
            for thre_s in np.where(np.array(scores) >= data_confidence_level)[0]:
                print(thre_s)
                tempo_l.append(predict_bbox[thre_s] / 300)
            predict_bbox_list.append(tempo_l)

            fig = self.vis_bbox(
                rgb_img,
                bbox=predict_bbox,
                label_index=pre_dict_label_index,
                scores=scores,
                label_names=self.eval_categories,
            )
            if save:
                fig.savefig(f"/{img_file_path}/Cygnus_{bat}.png")
        #             plt.imshow(fig)
        return predict_bbox_list

    def ssd_predict(self, image, width, height, data_confidence_level=0.5):
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

        torch_img_ = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        self.net.eval()  # ネットワークを推論モードへ
        with torch.no_grad():
            outputs, decoded_box = self.net(torch_img_)
            detections = self.detect(*outputs).to("cpu").detach().numpy().copy()
            # detectionsの形は、torch.Size([1, 21, 200, 5])  ※200はtop_kの値

        for i in range(1, 2):
            for k in range(0, 20):
                print(detections[0, i][k])

        # confidence_levelが基準以上を取り出す
        predict_bbox, pre_dict_label_index, scores = [], [], []

        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
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
        return predict_bbox, pre_dict_label_index, scores

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

        num_classes = len(label_names)  # クラス数（背景のぞく）
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 画像の表示
        fig = plt.figure(figsize=(5, 5), tight_layout=True)
        currentAxis = fig.add_subplot(111)
        currentAxis.imshow(rgb_img)

        # BBox分のループ
        for i, bb in enumerate(bbox):
            # ラベル名
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

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
            currentAxis.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=4))
            # 長方形の枠の左上にラベルを描画する
            currentAxis.text(xy[0], xy[1], display_txt, bbox={"facecolor": color, "alpha": 0.5}, fontsize=15)
        currentAxis.axis("off")
        plt.show()
        return fig
