import argparse
import glob
import os
import sys

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from PIL import Image
from sklearn.cluster import KMeans

sys.path.append("../")
from utils.ssd_model import SSD

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
    np.random.seed(123)
    torch.manual_seed(123)
    # cuDNN用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if len(args.NonRing_dir.split("/")[-1]) == 0:
        savedir_name = "/".join(args.NonRing_dir.split("/")[:-2]) + "/clustering_result"
    else:
        savedir_name = "/".join(args.NonRing_dir.split("/")[:-1]) + "/clustering_result"
    os.makedirs(savedir_name, exist_ok=True)

    print("Loading Model....")
    net_weights = torch.load(args.model_checkpoint)
    net_w = SSD()
    net_w.load_state_dict(net_weights["model_state_dict"])
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
    for i in tqdm.tqdm(path_list):
        data.append(np.array(Image.open(i)))
    data = np.array(data)[:, :, :, :2]

    ####################################################
    ## 画像をモデルに入力し、得られた特徴量をクラスタリングする ##
    ####################################################
    print("Clustering")

    features_list = []
    batch = np.linspace(0, data.shape[0], 500)
    ## データをモデルに入力する
    for i in tqdm.tqdm(range(len(batch) - 1)):
        x = data[int(batch[i]) : int(batch[i + 1])]
        p_data = torch.from_numpy(x) / 255
        p_data = p_data.permute(0, 3, 1, 2)

        with torch.no_grad():
            net_w.eval()
            p_data = p_data.to(device, dtype=torch.float)
            for k in range(51):
                p_data = net_w[k](p_data)
            features_list.append(p_data.to("cpu").detach().numpy().copy())

    ## 特徴量をクラスタリング
    features_list = np.concatenate(features_list)
    np.save(savedir_name + "/features_list.npy", features_list)
    prediction = KMeans(n_clusters=int(args.class_num), random_state=123, n_init="auto").fit_predict(
        features_list.reshape(features_list.shape[0], -1)
    )
    print("Clustering is done")
    print(
        f"Class 0 : {sum(prediction == 0)}\nClass 1 : {sum(prediction == 1)}\nClass 2 : {sum(prediction == 2)}\nClass 3 : {sum(prediction == 3)}\nClass 4 : {sum(prediction == 4)}\nClass 5 : {sum(prediction == 5)}\nClass 6 : {sum(prediction == 6)}\nClass 7 : {sum(prediction == 7)}\nClass 8 : {sum(prediction == 8)}"
    )

    ######################################
    ## Non-Ringデータのうち特定クラスを削除 ##
    ######################################

    print("Delete photo")
    for path, pred in zip(path_list, prediction):
        photo = path
        json = path[:-4] + ".json"
        if int(pred) == 1:
            os.remove(photo)
            os.remove(json)
        elif int(pred) == 4:
            os.remove(photo)
            os.remove(json)

    ##############
    ## 画像の作成 ##
    ##############
    fig, ax = plt.subplots()
    ax.hist(prediction, bins=int(args.class_num))
    ax.set_xlabel("クラス数", size=15)
    ax.set_ylabel("個数", size=15)
    fig.savefig(savedir_name + "/class_detail.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
