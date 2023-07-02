import glob
import json
import os
import shutil
import tarfile

import numpy as np
from numpy.random import default_rng
from PIL import Image
from sklearn.model_selection import ShuffleSplit

from make_Ring_data import make_ring
from sub import print_and_log


class make_training_val_data:
    """学習に使用するRraining_data, Validation dataを作成する。

    Params:
        augmentation_name (str): 保存するためのdirectory
        f_log (txt): logを保存するファイル
        args (args): 学習時に指定するargs

    :注意書き:
    >>> NonRingは毎回作成は高コストなため、事前に作成して、copyする。
    >>> 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない。
    """

    def __init__(self, augmentation_name, f_log, args):
        self.Data_rg = default_rng(args.fits_random_state)
        self.augmentation_name = augmentation_name
        self.f_log = f_log
        self.args = args
        # fmt: off
        fits_name = [
            "spitzer_00300+0000_rgb", "spitzer_00600+0000_rgb", "spitzer_00900+0000_rgb", "spitzer_01200+0000_rgb",
            "spitzer_01500+0000_rgb", "spitzer_01800+0000_rgb", "spitzer_02100+0000_rgb", "spitzer_02400+0000_rgb",
            "spitzer_02700+0000_rgb", "spitzer_03000+0000_rgb", "spitzer_03300+0000_rgb", "spitzer_03600+0000_rgb",
            "spitzer_03900+0000_rgb", "spitzer_04200+0000_rgb", "spitzer_04500+0000_rgb", "spitzer_04800+0000_rgb",
            "spitzer_05100+0000_rgb", "spitzer_05400+0000_rgb", "spitzer_05700+0000_rgb", "spitzer_06000+0000_rgb",
            "spitzer_29700+0000_rgb", "spitzer_30000+0000_rgb", "spitzer_30300+0000_rgb", "spitzer_30600+0000_rgb",
            "spitzer_30900+0000_rgb", "spitzer_31200+0000_rgb", "spitzer_31500+0000_rgb", "spitzer_31800+0000_rgb",
            "spitzer_32100+0000_rgb", "spitzer_32400+0000_rgb", "spitzer_32700+0000_rgb", "spitzer_33000+0000_rgb",
            "spitzer_33300+0000_rgb", "spitzer_33600+0000_rgb", "spitzer_33900+0000_rgb", "spitzer_34200+0000_rgb",
            "spitzer_34500+0000_rgb", "spitzer_34800+0000_rgb", "spitzer_35100+0000_rgb", "spitzer_35400+0000_rgb",
            "spitzer_35700+0000_rgb",
        ]
        # fmt: on

        if args.region_suffle:
            ss = ShuffleSplit(n_splits=args.n_splits, random_state=args.fits_random_state)
            train_index, val_index = list(ss.split(list(range(len(fits_name)))))[args.fits_index]
            self.train_l = [fits_name[i] for i in sorted(train_index)]
            self.val_l = [fits_name[i] for i in sorted(val_index)]
            print_and_log(
                self.f_log,
                [
                    "This training is shuffled Train region ",
                    "#################",
                    "  train_region",
                    "#################",
                    str(self.train_l),
                    " ",
                    "#################",
                    "   val_region",
                    "#################",
                    str(self.val_l),
                    " ",
                ],
            )

        else:
            ## 'spitzer_29400+0000_rgb'は、8µmのデータが全然ないため使用しない
            # fmt: off
            train_l = [
                "spitzer_02100+0000_rgb", "spitzer_04200+0000_rgb", "spitzer_33300+0000_rgb", "spitzer_35400+0000_rgb",
                "spitzer_00300+0000_rgb", "spitzer_02400+0000_rgb", "spitzer_04500+0000_rgb", "spitzer_31500+0000_rgb",
                "spitzer_33600+0000_rgb", "spitzer_35700+0000_rgb", "spitzer_00600+0000_rgb", "spitzer_02700+0000_rgb",
                "spitzer_04800+0000_rgb", "spitzer_29700+0000_rgb", "spitzer_31800+0000_rgb", "spitzer_03000+0000_rgb",
                "spitzer_05100+0000_rgb", "spitzer_30000+0000_rgb", "spitzer_32100+0000_rgb", "spitzer_01200+0000_rgb",
                "spitzer_03300+0000_rgb", "spitzer_05400+0000_rgb", "spitzer_30300+0000_rgb", "spitzer_32400+0000_rgb",
                "spitzer_34500+0000_rgb", "spitzer_01500+0000_rgb", "spitzer_03600+0000_rgb", "spitzer_05700+0000_rgb",
                "spitzer_30600+0000_rgb", "spitzer_32700+0000_rgb", "spitzer_34800+0000_rgb", "spitzer_01800+0000_rgb",
                "spitzer_06000+0000_rgb", "spitzer_30900+0000_rgb", "spitzer_33000+0000_rgb", "spitzer_35100+0000_rgb"
                ]
            # fmt: on
            self.train_l = sorted(train_l)

        ## 必要なフォルダの作成
        self.save_data_path = args.savedir_path + "".join("dataset") + "/" + augmentation_name.split("/")[-1]
        os.makedirs(self.save_data_path, exist_ok=True)
        os.makedirs(self.args.savedir_path + "".join("dataset"), exist_ok=True)
        os.makedirs(self.save_data_path + "/train", exist_ok=True)
        os.makedirs(self.save_data_path + "/train/ring", exist_ok=True)
        os.makedirs(self.save_data_path + "/train/nonring", exist_ok=True)

    def make_training_data(self, train_cfg):
        """Trainingデータを作成する関数。

        Params:
            train_cfg (list):どのaugmentationを使用するか
        """
        ## train_dataのshapeは、(Num, 300, 300, 3)/ typeはfloat32型
        self.train_data, train_label = make_ring(
            self.augmentation_name, train_cfg, self.args, self.train_l, self.Data_rg
        )

        ## Trainingデータをpngファイルに変換＋保存
        for i in range(self.train_data.shape[0]):
            pil_image = Image.fromarray(np.uint8(self.train_data[i] * 255))
            pil_image.save(f"{self.save_data_path}/train/ring/Ring_{i}.png")

        ## Training labelをjsonに変換＋保存
        for i, row in train_label.iterrows():
            ll = []
            if len(row["xmin"]) >= 1:
                for la in range(len(row["xmin"])):
                    ll.append(
                        {
                            "Confidence": str(0),
                            "XMin": str(row["xmin"][la]),
                            "XMax": str(row["xmax"][la]),
                            "YMin": str(row["ymin"][la]),
                            "YMax": str(row["ymax"][la]),
                        }
                    )
            else:
                pass

        with open(f"{self.save_data_path}/train/ring/Ring_{i}.json", "w") as f:
            json.dump(ll, f, indent=4)

        ########################################
        ## Trainingに用いるNon-Ringデータをコピー ##
        ########################################
        if self.args.region_suffle:
            ## 領域ごとのNonRingをcopyする。
            NonRing_origin = []
            _ = [glob.glob(f"/workspace/NonRing_png/region_NonRing_png/{i}/*.png") for i in self.train_l]
            [NonRing_origin.extend(i) for i in _]
            Choice_NonRing = self.Data_rg.choice(
                NonRing_origin, int(self.train_data.shape[0]) * self.args.NonRing_ratio, replace=False
            )
            for i, k in enumerate(Choice_NonRing):
                shutil.copyfile(k, f"{self.save_data_path}/train/nonring/NonRing_{i}.png")
                shutil.copyfile(k[:-3] + "json", f"{self.save_data_path}/train/nonring/NonRing_{i}.json")
        else:
            ## デフォルトのNonRingをcopyする。
            NonRing_origin = glob.glob("/workspace/NonRing_png/default_NonRing_png/train/*.png")
            Choice_NonRing = self.Data_rg.choice(
                NonRing_origin, int(self.train_data.shape[0]) * self.args.NonRing_ratio, replace=False
            )
            for i in Choice_NonRing:
                shutil.copyfile(i, "%s/train/nonring/%s" % (self.save_data_path, i.split("/")[-1]))
                shutil.copyfile(
                    i[:-3] + "json", "%s/train/nonring/%s" % (self.save_data_path, i.split("/")[-1][:-3] + "json")
                )

        with tarfile.open(f"{self.save_data_path}/bubble_dataset_train_ring.tar", "w:gz") as tar:
            tar.add(f"{self.save_data_path}/train/ring")

        with tarfile.open(f"{self.save_data_path}/bubble_dataset_train_nonring.tar", "w:gz") as tar:
            tar.add(f"{self.save_data_path}/train/nonring")

        return (
            f"{self.save_data_path}/bubble_dataset_train_ring.tar",
            f"{self.save_data_path}/bubble_dataset_train_nonring.tar",
        )

    def make_validation_data(self):
        """Validationに用いるRing / NonRingをコピーする。

        Params:
            train_cfg (_type_): _description_
        """

        ## ********* 各領域ごとに *********
        if self.args.region_suffle:
            os.makedirs(f"{self.save_data_path}/val", exist_ok=True)

            ## Ringデータをコピーする。
            Val_origin = []
            a = [glob.glob(f"/workspace/val_png/region_val_png/{i}/*.png") for i in self.val_l]
            [Val_origin.extend(i) for i in a]
            for i, k in enumerate(Val_origin):
                shutil.copyfile(k, f"{self.save_data_path}/val/Ring_{i}.png")
                shutil.copyfile(k[:-3] + "json", f"{self.save_data_path}/val/Ring_{i}.json")

            ## Non-Ringをコピーする
            NonRing_origin = []
            a = [glob.glob(f"/workspace/NonRing_png/region_NonRing_png/{i}/*.png") for i in self.val_l]
            [NonRing_origin.extend(i) for i in a]
            Choice_NonRing = self.Data_rg.choice(
                NonRing_origin, int(len(Val_origin)) * self.args.NonRing_ratio, replace=False
            )
            for i, k in enumerate(Choice_NonRing):
                shutil.copyfile(k, f"{self.save_data_path}/val/NonRing_{i}.png")
                shutil.copyfile(k[:-3] + "json", f"{self.save_data_path}/val/NonRing_{i}.json")

        ## ********* デフォルト領域で *********
        else:
            ## Validationデータをcopyする。
            ## Ringデータのコピー
            shutil.copytree("/workspace/val_png/default_val", f"{self.save_data_path}/val", dirs_exist_ok=True)

            ## Non-Ringデータのコピー
            Val_default_path = glob.glob("/workspace/NonRing_png/default_NonRing_png/val/*.png")
            Choice_NonRing = self.Data_rg.choice(
                Val_default_path,
                int(len(glob.glob(f"{self.save_data_path}/val/*"))) * self.args.NonRing_ratio,
                replace=False,
            )
            for i, k in enumerate(Choice_NonRing):
                shutil.copyfile(k, f"{self.save_data_path}/val/NonRing_{i}.png")
                shutil.copyfile(k[:-3] + "json", f"{self.save_data_path}/val/NonRing_{i}.json")

        with tarfile.open(f"{self.save_data_path}/bubble_dataset_val.tar", "w:gz") as tar:
            tar.add(f"{self.save_data_path}/val")

        return f"{self.save_data_path}/bubble_dataset_val.tar"

    def data_logger(self):
        ## TrainingとValidationの Ring & NonRing の枚数を取得
        train_Ring_num = len(glob.glob(f"{self.save_data_path}/train/Ring_*.json"))
        val_Ring_num = len(glob.glob(f"{self.save_data_path}/val/Ring_*.json"))
        Train_Non_Ring_num = len(glob.glob(f"{self.save_data_path}/train/nonring/NonRing_*.json"))
        Val_Non_Ring_num = len(glob.glob(f"{self.save_data_path}/val/NonRing_*.json"))

        ## logに記入
        if np.isnan(np.sum(self.train_data)):
            mg = "Training data include Nan"
        else:
            mg = "Training data dont include Nan"

        print_and_log(
            self.f_log,
            [
                "====================================",
                f"Ring NonRing ratio = 1 : {self.args.NonRing_ratio}",
                " ",
                "confirm nan in Training Data",
                f">>> Ring_data: {mg}",
                " ",
                "Ring & Non-Ring num",
                f">>> Train Ring num: {train_Ring_num}",
                f">>> Val Ring num: {val_Ring_num}",
                " ",
                f">>> Train Non-Ring num: {Train_Non_Ring_num}",
                f">>> Val Non-Ring num: {Val_Non_Ring_num}",
                " ",
                f">>> Total Train num: {train_Ring_num + Train_Non_Ring_num}",
                f">>> Total Val num: {val_Ring_num + Val_Non_Ring_num}",
            ],
        )

        return train_Ring_num
