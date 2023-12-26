import glob
import os
import shutil
import tarfile

import numpy as np
import torch
import webdataset
from data import od_collate_fn_validation, preprocess_validation
from training_sub import calc_fscore_val
from utils.ssd_model import SSD, Detect


def infer_test(model_path, args, val_size, val_best_confthre):
    """l18の推論を行う関数

    Args:
        model_path (str): path of the model to infer
        args (args)     : argparse

    Returns:
        f1_score (float)      : F1 score
        precision (float)     : precision
        recall (float)        : recall
        conf_threshold (float): conf_threshold
    """
    result, position, regions = [], [], []
    save_data_path = args.savedir_path + "".join("dataset") + "/" + model_path.split("/")[-1]
    os.makedirs(f"{save_data_path}/test", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detect = Detect(nms_thresh=0.45, top_k=500, conf_thresh=0.3)

    ################
    ## データの作成 ##
    ################
    if os.path.exists(f"{save_data_path}/bubble_dataset_test.tar"):
        os.remove(f"{save_data_path}/bubble_dataset_test.tar")

    test_region = [
        "spitzer_01200+0000_rgb",
        "spitzer_01500+0000_rgb",
        "spitzer_01800+0000_rgb",
        "spitzer_02100+0000_rgb",
    ]
    test_path = []
    for region in test_region:
        for size in val_size:
            test_path.extend(glob.glob(f"{args.validation_data_path}/{region}/*/*_{size}_*.png"))
    for k in test_path:
        shutil.copyfile(k, f"{save_data_path}/test/{k.split('/')[-1][:-4]}.png")
        shutil.copyfile(k[:-3] + "json", f"{save_data_path}/test/{k.split('/')[-1][:-4]}.json")

    with tarfile.open(f"{save_data_path}/bubble_dataset_test.tar", "w:gz") as tar:
        tar.add(f"{save_data_path}/test")

    Dataset_test = (
        webdataset.WebDataset(f"{save_data_path}/bubble_dataset_test.tar")
        .decode("pil")
        .to_tuple("png", "json", "__key__")
        .map(preprocess_validation)
    )
    dl_test = torch.utils.data.DataLoader(
        Dataset_test,
        collate_fn=od_collate_fn_validation,
        batch_size=args.Val_mini_batch,
        num_workers=2,
        pin_memory=True,
    )

    ##############
    ## l18の推論 ##
    ##############
    net = SSD()
    net_weights = torch.load(model_path + "/earlystopping.pth")
    net.load_state_dict(net_weights["model_state_dict"])
    net.to(device)
    net.eval()
    all_iter = len(test_path) / args.Val_mini_batch
    iteration = 0

    for _ in dl_test:
        images, targets, offset, region_info = _[0], _[1], _[2], _[3]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)[:, :2, :, :]
        images = images.to(device, dtype=torch.float)
        targets = [ann.to(device, dtype=torch.float) for ann in targets]

        with torch.no_grad():
            outputs, decoded_box = net(images)
            print("\r" + str(iteration) + "/" + str(all_iter) + " ", end="")
            iteration += 1
            result.append(detect(*outputs).to("cpu").detach().numpy().copy())
            position.extend(offset)
            regions.extend(region_info)

    f_score, precision, recall, conf_threshold = calc_fscore_val(
        np.concatenate(result),
        np.array(position),
        regions,
        args,
        threshold=val_best_confthre,
        save=True,
        save_path=model_path,
    )

    return f_score, precision, recall, conf_threshold
