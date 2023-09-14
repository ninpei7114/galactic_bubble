import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import japanize_matplotlib

sys.path.append("../")
from processing import data_view_rectangl

"""
Non-Ringをクラスタリングするためのスクリプト

example command:
python confirm_clustering.py /workspace/NonRing_png/region_NonRing_png /workspace/NonRing_png/clustering_result/features_list.npy
"""


def parse_args():
    ## クラス数、モデルのcheckpoint、Non-Ring画像の場所
    parser = argparse.ArgumentParser(description="make data for SSD")
    parser.add_argument("NonRing_dir", help="NonRing画像の場所")
    parser.add_argument("features_list", help="NonRing画像の場所")

    return parser.parse_args()


def main(args):
    print("Loading data....")
    ## データの取得と形成
    path_list = sorted(glob.glob("%s/*/*.png" % args.NonRing_dir))
    data = []
    for i in path_list:
        data.append(np.array(Image.open(i)))
    data = np.array(data)

    for class_num in range(2, 10):
        features_list = np.load(args.features_list)
        os.makedirs("/".join(args.NonRing_dir.split("/")[:-1]) + f"/clustering_result/class{class_num}", exist_ok=True)
        prediction = KMeans(n_clusters=int(class_num), random_state=123, n_init="auto").fit_predict(
            features_list.reshape(features_list.shape[0], -1)
        )
        fig, ax = plt.subplots()
        ax.hist(prediction, bins=int(class_num))
        ax.set_xlabel("クラス数", size=15)
        ax.set_ylabel("個数", size=15)
        fig.savefig(
            "/".join(args.NonRing_dir.split("/")[:-1]) + f"/clustering_result/class{class_num}/class_detail.png"
        )

        num_list = []
        for k in range(int(class_num)):
            slice_ = int(np.sum(prediction == k) / 90)
            data_view_rectangl(25, data[prediction == k][::slice_]).save(
                "/".join(args.NonRing_dir.split("/")[:-1]) + f"/clustering_result/class{class_num}/clus_{k}.png"
            )
            num_list.append(np.sum(prediction == k))
        df = pd.DataFrame(num_list).T
        df.to_csv("/".join(args.NonRing_dir.split("/")[:-1]) + f"/clustering_result/class{class_num}/class_num.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
