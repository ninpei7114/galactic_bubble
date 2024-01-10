import numpy as np


def od_collate_fn_validation(batch):
    imgs, offset, region_info = [], [], []
    for sample in batch:
        imgs.append(sample[0])
        offset.append(
            [
                sample[1].split("/")[-1].split("_")[-4],
                sample[1].split("/")[-1].split("_")[-3],
                sample[1].split("/")[-1].split("_")[-2],
            ]
        )
        region_info.append(sample[1].split("/")[-1].split("_")[-1])
    imgs = np.array(imgs)
    offset = np.array(offset)

    return imgs, offset, region_info


## webdatasetのためのval_preprocess
def preprocess_validation(sample):
    img, key = sample
    return (np.array(img) / 255, key)
