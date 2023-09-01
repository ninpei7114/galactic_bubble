import os
import glob

import numpy as np
import torch
import tarfile
import shutil
import webdataset

from utils.ssd_model import Detect
from utils.ssd_model import SSD
from data import preprocess_validation, od_collate_fn_validation
from training_sub import calc_f1score_val


def infer_l18(model_path, args):
    result, position, regions = [], [], []
    save_data_path = args.savedir_path + "".join("dataset") + "/" + model_path.split("/")[-1]
    os.makedirs(f"{save_data_path}/l18", exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detect = Detect(nms_thresh=0.45, top_k=500, conf_thresh=0.3)

    ################
    ## データの作成 ##
    ################
    if os.path.exists(f"{save_data_path}/bubble_dataset_l18.tar"):
        pass
    else:
        l18_path = glob.glob(f"{args.validation_data_path}/spitzer_01800+0000_rgb/*/*.png")
        for k in l18_path:
            shutil.copyfile(k, f"{save_data_path}/l18/{k.split('/')[-1][:-4]}.png")
            shutil.copyfile(k[:-3] + "json", f"{save_data_path}/l18/{k.split('/')[-1][:-4]}.json")

        with tarfile.open(f"{save_data_path}/bubble_dataset_l18.tar", "w:gz") as tar:
            tar.add(f"{save_data_path}/l18")

    Dataset_l18 = (
        webdataset.WebDataset(f"{save_data_path}/bubble_dataset_l18.tar")
        .decode("pil")
        .to_tuple("png", "json", "__key__")
        .map(preprocess_validation)
    )
    dl_l18 = torch.utils.data.DataLoader(
        Dataset_l18,
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
    all_iter = len(glob.glob(f"{args.validation_data_path}/spitzer_01800+0000_rgb/*/*.png")) / args.Val_mini_batch
    iteration = 0

    for _ in dl_l18:
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

    f1_score, precision, recall, conf_threshold = calc_f1score_val(
        np.concatenate(result), np.array(position), regions, args
    )

    return f1_score, precision, recall, conf_threshold
