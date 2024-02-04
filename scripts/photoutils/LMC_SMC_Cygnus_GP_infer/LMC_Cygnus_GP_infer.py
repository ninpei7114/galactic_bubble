import argparse
import glob
import os
import sys
import tarfile
import time

import numpy as np
import torch
import wandb
import webdataset

sys.path.append("/home/cygnus/jupyter/galactic_bubble/scripts/photoutils")
from infer_sub import od_collate_fn_validation, preprocess_validation
from utils.ssd_model import SSD, Detect

"""Example command line:

python LMC_Cygnus_GP_infer.py galactic_bubble/clustering_NewNorm/training_log:v0
"""


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Implementation of SSD")
    parser.add_argument("model_ver", type=str, help="model's path to infer")
    parser.add_argument("result_save_dir", type=str, help="Infer Result Save Directory")
    parser.add_argument("Cygnus_LMC_SMC_png", type=str, help="Cygnus_LMC_SMC_png")

    return parser.parse_args()


def main(args):
    start = time.time()
    torch.backends.cudnn.benchmark = True
    device = torch.device(torch.device("cuda:0") if torch.cuda.is_available() else "cpu")

    detect = Detect(nms_thresh=0.3, top_k=1000)
    model_ver = "/".join(args.model_ver.split("/")[-2:])
    os.makedirs(f"{args.result_save_dir}/{model_ver}", exist_ok=True)
    f_log = open(f"{args.result_save_dir}/{model_ver}/" + "/log.txt", "w")
    f_log.write("使用モデル: " + args.model_ver + "\n")
    f_log.close()

    model_download_dir = f"{args.result_save_dir}/artifacts/"
    api = wandb.Api()
    artifact = api.artifact(f"{args.model_ver}")
    artifact.download(model_download_dir + "/".join(args.model_ver.split("/")[-2:]))
    net_w = SSD()
    net_weights = torch.load(model_download_dir + "/".join(args.model_ver.split("/")[-2:]) + "/earlystopping.pth")
    net_w.load_state_dict(net_weights["model_state_dict"])
    net_w.to(device)
    net_w.eval()

    for region in ["LMC", "Cygnus", "SMC"]:
        print(f"\n{region=}")
        tarfile_name = f"{args.Cygnus_LMC_SMC_png}/{region}_dataset.tar"
        if not os.path.exists(tarfile_name):
            with tarfile.open(tarfile_name, "w:gz") as tar:
                tar.add(f"{args.Cygnus_LMC_SMC_png}/{region}")

        Dataset_test = (
            webdataset.WebDataset(tarfile_name).decode("pil").to_tuple("png", "__key__").map(preprocess_validation)
        )
        dl_region = torch.utils.data.DataLoader(
            Dataset_test,
            collate_fn=od_collate_fn_validation,
            batch_size=128,
            num_workers=2,
            pin_memory=True,
        )
        all_iter = int(len(glob.glob(f"{args.Cygnus_LMC_SMC_png}/{region}/*.png")) / 128)
        iteration = 0
        ################
        ## INFER PART ##
        ################
        position, result, regions = [], [], []
        print("START INFER")
        for _ in dl_region:
            images, offset, region_info = _[0], _[1], _[2]
            images = torch.from_numpy(images).permute(0, 3, 1, 2)[:, :2, :, :]
            images = images.to(device, dtype=torch.float)

            with torch.no_grad():
                outputs, _ = net_w(images)
                print("\r" + str(iteration) + "/" + str(all_iter) + " ", end="")
                iteration += 1
                result.append(detect(*outputs).to("cpu").detach().numpy().copy().astype(np.float32))
                position.extend(offset)
                regions.extend(region_info)

        position = np.array(position)
        result = np.concatenate(result)
        os.makedirs(f"{args.result_save_dir}/{model_ver}/{region}", exist_ok=True)
        np.save(f"{args.result_save_dir}/{model_ver}/{region}/position.npy", position)
        np.save(f"{args.result_save_dir}/{model_ver}/{region}/result.npy", result)

    print(f"elapsed_time:{time.time() - start}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
