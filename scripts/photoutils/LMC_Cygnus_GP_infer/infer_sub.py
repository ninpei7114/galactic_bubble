import numpy as np
import torch


def od_collate_fn_validation(batch):
    targets = []
    imgs = []
    offset = []
    region_info = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        offset.append(
            [
                sample[2].split("/")[-1].split("_")[-4],
                sample[2].split("/")[-1].split("_")[-3],
                sample[2].split("/")[-1].split("_")[-2],
            ]
        )
        region_info.append(sample[2].split("/")[-1].split("_")[-1])
    imgs = np.array(imgs)
    offset = np.array(offset)

    return imgs, targets, offset, region_info


## webdatasetのためのval_preprocess
def preprocess_validation(sample):
    img, key = sample
    return (np.array(img) / 255, key)
