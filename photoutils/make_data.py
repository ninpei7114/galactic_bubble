import glob
import json
import os
import shutil
import tarfile

import numpy as np
from make_Ring_data import make_ring
from numpy.random import default_rng
from sklearn.model_selection import ShuffleSplit
from training_sub import print_and_log


class make_training_val_data:
    """Creates Training_data and Validation data for use in learning.

    Params:
        augmentation_name (str): Directory to save
        f_log (txt): File to save the log
        args (args): Args to specify during learning

    :注意書き:
    >>> NonRing is costly to create every time, so it is created in advance and copied.
    >>> 'spitzer_29400+0000_rgb' is not used because there is hardly any 8µm data.
    >>> "spitzer_01200+0000_rgb", "spitzer_01500+0000_rgb", "spitzer_01800+0000_rgb", "spitzer_02100+0000_rgb" are not used because they are test region.
    """

    def __init__(self, augmentation_name, f_log, args):
        self.Data_rg = default_rng(args.data_random_state)
        self.augmentation_name = augmentation_name
        self.f_log = f_log
        self.args = args
        # fmt: off
        fits_name = [
            "spitzer_00300+0000_rgb", "spitzer_00600+0000_rgb", "spitzer_00900+0000_rgb", "spitzer_02400+0000_rgb",
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

        # if args.region_suffle:
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

        ## Create necessary folders
        self.save_data_path = args.savedir_path + "/" + "".join("dataset") + "/" + augmentation_name.split("/")[-1]
        os.makedirs(self.save_data_path, exist_ok=True)
        os.makedirs(self.args.savedir_path + "/" + "".join("dataset"), exist_ok=True)

    def make_training_ring_data(self, train_cfg, epoch):
        """Function to create training data.

        Params:
            train_cfg (list) : List of augmentations to use.
            epoch (int)      : how many epochs
        """
        ## The shape of train_data is (Num, 300, 300, 3) with data type float32
        os.makedirs(self.save_data_path + "/train", exist_ok=True)
        os.makedirs(self.save_data_path + "/train/ring", exist_ok=True)
        make_ring(self.augmentation_name, train_cfg, self.args, self.train_l, self.Data_rg, epoch, self.save_data_path)

        with tarfile.open(f"{self.save_data_path}/bubble_dataset_train_ring.tar", "w:gz") as tar:
            tar.add(f"{self.save_data_path}/train/ring")

        return self.save_data_path

    def make_training_nonring_data(self):
        print("MAKE NONRING DATA ...")
        os.makedirs(self.save_data_path + "/train/nonring", exist_ok=True)

        #####################################
        ## Copy Non-Ring data for training ##
        #####################################
        # NonRing_num_list = []

        ## Copy NonRing data for each region.
        ## Copy NonRing for each Non-Ring class
        NonRing_path = []
        _ = [glob.glob(f"{self.args.NonRing_data_path}/{i}/*.png") for i in self.train_l]
        [NonRing_path.extend(i) for i in _]
        # NonRing_num_list.append(len(NonRing_path))
        for i, k in enumerate(NonRing_path):
            shutil.copyfile(k, f"{self.save_data_path}/train/nonring/NonRing_{i}.png")
            shutil.copyfile(k[:-3] + "json", f"{self.save_data_path}/train/nonring/NonRing_{i}.json")

        # for cl in NonRing_class_num:
        with tarfile.open(f"{self.save_data_path}/bubble_dataset_train_nonring.tar", "w:gz") as tar:
            tar.add(f"{self.save_data_path}/train/nonring/")

        return len(NonRing_path)

    def make_validation_data(self, val_size):
        """Copy the Ring / NonRing to be used for Validation.

        Params:
            train_cfg (_type_): _description_
        """
        os.makedirs(f"{self.save_data_path}/val", exist_ok=True)
        print("MAKE VALIDATION DATA ...")

        ## Copy the Ring data.
        Val_origin = []
        for i in self.val_l:
            for size in val_size:
                a = glob.glob(f"{self.args.validation_data_path}/{i}/*/*_{size}_*.png")
                Val_origin.extend(a)

        for k in Val_origin:
            shutil.copyfile(k, f"{self.save_data_path}/val/{k.split('/')[-1][:-4]}.png")
            shutil.copyfile(k[:-3] + "json", f"{self.save_data_path}/val/{k.split('/')[-1][:-4]}.json")

        ## Convert to tar file
        with tarfile.open(f"{self.save_data_path}/bubble_dataset_val.tar", "w:gz") as tar:
            tar.add(f"{self.save_data_path}/val")

        return f"{self.save_data_path}/bubble_dataset_val.tar", len(glob.glob(f"{self.save_data_path}/val/*.png"))

    def data_logger(self):
        """Obtain the number of Ring & NonRing used for Training and Validation.

        Returns(int): The number of Ring used for Training
        """
        train_Ring_num = len(glob.glob(f"{self.save_data_path}/train/ring/Ring_*.json"))
        val_Ring_num = len(glob.glob(f"{self.save_data_path}/val/Ring_*.json"))
        Train_Non_Ring_num = len(glob.glob(f"{self.save_data_path}/train/nonring/NonRing_*.json"))
        Val_Non_Ring_num = len(glob.glob(f"{self.save_data_path}/val/NonRing_*.json"))

        print_and_log(
            self.f_log,
            [
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
