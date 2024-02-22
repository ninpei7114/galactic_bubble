import argparse
import glob
import os
import shutil
import sys

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
import wandb
from PIL import Image
from sklearn.cluster import KMeans

sys.path.append("../")
from processing import data_view_rectangl
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
    parser.add_argument("model_ver", help="特徴量を生成するためのモデルのcheckpointの場所")
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
        model_download_dir = "/".join(args.NonRing_dir.split("/")[:-2]) + "/model"
    else:
        savedir_name = "/".join(args.NonRing_dir.split("/")[:-1]) + "/clustering_result"
        model_download_dir = "/".join(args.NonRing_dir.split("/")[:-1]) + "/model"
    os.makedirs(savedir_name, exist_ok=True)
    os.makedirs(model_download_dir, exist_ok=True)

    api = wandb.Api()
    artifact = api.artifact(f"{args.model_ver}")
    artifact.download(model_download_dir)

    print("Loading Model....")
    net_weights = torch.load(model_download_dir + "/earlystopping.pth")
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
    for i in range(int(args.class_num)):
        print(f"Class {i} : {sum(prediction == i)}")

    #################################
    ## Non-Ringデータをクラスごとに移動##
    #################################
    for i in glob.glob("%s/*" % args.NonRing_dir):
        for k in range(int(args.class_num)):
            os.makedirs("%s/class%s" % (i, k), exist_ok=True)

    print("Move photo")
    for path, pred in zip(path_list, prediction):
        photo = path
        json = path[:-4] + ".json"

        shutil.move(photo, "/".join(photo.split("/")[:-1]) + f"/class{pred}/" + photo.split("/")[-1])
        shutil.move(json, "/".join(json.split("/")[:-1]) + f"/class{pred}/" + json.split("/")[-1])

    ##############
    ## 画像の作成 ##
    ##############
    fig, ax = plt.subplots()
    ax.hist(prediction, bins=int(args.class_num))
    ax.set_xlabel("クラス数", size=15)
    ax.set_ylabel("個数", size=15)
    fig.savefig(savedir_name + "/class_detail.jpeg")

    num_list = []
    for k in range(int(args.class_num)):
        class_path = glob.glob(args.NonRing_dir + f"/*/class{k}/*.png")
        slice_ = int(len(class_path) / 90)
        class_data_list = []
        for i in class_path[::slice_]:
            class_data_list.append(np.array(Image.open(i)))
        data_view_rectangl(25, np.array(class_data_list)).save(savedir_name + f"/clus_{k}.jpeg")
        num_list.append(len(class_path))
    df = pd.DataFrame(num_list).T
    df.to_csv(savedir_name + "/class_num.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
