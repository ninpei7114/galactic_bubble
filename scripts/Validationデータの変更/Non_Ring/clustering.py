import torch
import torch.nn as nn

import numpy as np
from numpy.random import default_rng
from PIL import Image
from sklearn.cluster import KMeans
from skimage import transform

import sys

sys.path.append("../")
from utils.ssd_model import SSD

import argparse
import glob
import tqdm
import copy
import shutil


"""
Non-Ringをクラスタリングするためのスクリプト

example command:
python clustering.py 7 /workspace/earlystopping.pth /workspace/NonRing_png/region_NonRing_png
"""


def parse_args():
    ## クラス数、モデルのcheckpoint、Non-Ring画像の場所
    parser = argparse.ArgumentParser(description="make data for SSD")
    parser.add_argument("class_num", help="clusteringの個数 (default: 8)", default=8)
    parser.add_argument("model_checkpoint", help="特徴量を生成するためのモデルのcheckpointの場所")
    parser.add_argument("NonRing_dir", help="NonRing画像の場所")

    return parser.parse_args()


def main(args):
    #################################
    ## クラスタリングのモデルを読み込む ##
    #################################

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading Model....")
    net_weights = torch.load(args.model_checkpoint)

    input_size = 300

    ssd_cfg = {
        "num_classes": 2,  # 背景クラスを含めた合計クラス数
        "input_size": input_size,  # 画像の入力サイズ
        "bbox_aspect_num": [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
        "feature_maps": [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        "steps": [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        "min_sizes": [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        "max_sizes": [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        "aspect_ratios": [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    }

    net_w = SSD(cfg=ssd_cfg)
    net_w.load_state_dict(net_weights)

    ## vggとextraで構成されるモデルを構築
    ## 特徴量サイズ、1x1x256
    for i in net_w.extras:
        net_w.vgg.append(i)
        net_w.vgg.append(nn.ReLU())

    net_w = net_w.vgg
    net_w.to(device)

    print("Loading data....")
    ## データの取得と形成
    path_list = sorted(glob.glob("%s/*/*.png" % args.NonRing_dir))
    data = []
    for i in path_list:
        data.append(np.array(Image.open(i)))

    data = np.array(data)[:, :, :, :2]
    features_list = []

    ####################################################
    ## 画像をモデルに入力し、得られた特徴量をクラスタリングする ##
    ####################################################

    print("Clustering")
    batch = np.linspace(0, data.shape[0], int(data.shape[0] / 32))

    ## データをモデルに入力する
    for i in tqdm.tqdm(range(len(batch) - 1)):
        x = data[int(batch[i]) : int(batch[i + 1])]
        p_data = torch.from_numpy(x)
        p_data = p_data.permute(0, 3, 1, 2)

        with torch.no_grad():
            net_w.eval()
            p_data = p_data.to(device, dtype=torch.float)
            for k in range(51):
                p_data = net_w[k](p_data)
            features_list.append(p_data.to("cpu").detach().numpy().copy())

    ## 特徴量をクラスタリング
    features_list = np.concatenate(features_list)
    prediction = KMeans(n_clusters=int(args.class_num), random_state=123).fit_predict(
        features_list.reshape(features_list.shape[0], -1)
    )

    ####################################
    ## Non-Ringデータをクラスごとに分ける ##
    ####################################
    NonRing_random_uni = default_rng(123)
    count = 0
    for path, pred in zip(path_list, prediction):
        photo = path
        json = path[:-4] + ".json"

        if pred == 3:
            photo_data = np.array(Image.open(photo))

            for aug in NonRing_random_uni.integers(0, 2, 2):
                if aug == 0:
                    tempo = copy.deepcopy(photo_data)
                    ud = np.flipud(tempo)
                    count += 1
                elif aug == 1:
                    lr = np.fliplr(tempo)
                else:
                    rotate_cut_data = transform.rotate(tempo, NonRing_random_uni.integers(0, [90, 180, 270], 1))

        shutil.move(photo, "/".join(photo.split("/")[:-1]) + "/class%s" % pred)
        shutil.move(json, "/".join(photo.split("/")[:-1]) + "/class%s" % pred)


if __name__ == "__main__":
    args = parse_args()
    main(args)
