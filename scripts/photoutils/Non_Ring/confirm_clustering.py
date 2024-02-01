import argparse
import glob
import os
import sys

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from PIL import Image
from sklearn.cluster import KMeans

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
    if len(args.NonRing_dir.split("/")[-1]) == 0:
        savedir_name = "/".join(args.NonRing_dir.split("/")[:-2]) + "/clustering_result"
    else:
        savedir_name = "/".join(args.NonRing_dir.split("/")[:-1]) + "/clustering_result"
    print("Loading data....")
    ## データの取得と形成
    path_list = sorted(glob.glob("%s/*/*.png" % args.NonRing_dir))
    data = []
    for i in tqdm.tqdm(path_list):
        data.append(np.array(Image.open(i)))
    data = np.array(data)

    for class_num in range(2, 10):
        features_list = np.load(args.features_list)
        os.makedirs(savedir_name + f"/class{class_num}", exist_ok=True)
        print("======================================")
        print(f"start clustering by {class_num}....")
        prediction = KMeans(n_clusters=int(class_num), random_state=123, n_init="auto").fit_predict(
            features_list.reshape(features_list.shape[0], -1)
        )
        print("Clustering is done")
        print(
            f"Class 0 : {sum(prediction == 0)}\nClass 1 : {sum(prediction == 1)}\nClass 2 : {sum(prediction == 2)}\nClass 3 : {sum(prediction == 3)}\nClass 4 : {sum(prediction == 4)}\nClass 5 : {sum(prediction == 5)}\nClass 6 : {sum(prediction == 6)}\nClass 7 : {sum(prediction == 7)}\nClass 8 : {sum(prediction == 8)}"
        )
        fig, ax = plt.subplots()
        ax.hist(prediction, bins=int(class_num))
        ax.set_xlabel("クラス数", size=15)
        ax.set_ylabel("個数", size=15)
        fig.savefig(savedir_name + f"/class{class_num}/class_detail.jpeg")

        num_list = []
        for k in range(int(class_num)):
            slice_ = int(np.sum(prediction == k) / 80)
            data_view_rectangl(25, data[prediction == k][::slice_]).save(
                savedir_name + f"/class{class_num}/clus_{k}.jpeg"
            )
            num_list.append(np.sum(prediction == k))
        df = pd.DataFrame(num_list).T
        df.to_csv(savedir_name + f"/class{class_num}/class_num.csv")


if __name__ == "__main__":
    args = parse_args()
    main(args)
