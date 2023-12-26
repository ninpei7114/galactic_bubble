import glob
from itertools import product as product
from math import sqrt as sqrt

import numpy as np
import torch
import webdataset


## webdatasetのために作成
def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    imgs = np.array(imgs)

    return imgs, targets


## webdatasetのために作成
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
                sample[2].split("/")[-1].split("_")[2],
                sample[2].split("/")[-1].split("_")[3],
                sample[2].split("/")[-1].split("_")[4],
            ]
        )
        region_info.append(sample[2].split("/")[-1].split("_")[5])
    imgs = np.array(imgs)
    offset = np.array(offset)

    return imgs, targets, offset, region_info


## webdatasetのために作成
def preprocess(sample):
    img, json = sample
    return np.array(img) / 255, [
        (float(x["XMin"]), float(x["YMin"]), float(x["XMax"]), float(x["YMax"]), float(x["Confidence"])) for x in json
    ]


## webdatasetのためのval_preprocess
def preprocess_validation(sample):
    img, json, key = sample
    return (
        np.array(img) / 255,
        [
            (float(x["XMin"]), float(x["YMin"]), float(x["XMax"]), float(x["YMax"]), float(x["Confidence"]))
            for x in json
        ],
        key,
    )


# 無限イテレータ
def InfiniteIterator(loader):
    iter = loader.__iter__()
    while True:
        try:
            x = next(iter)
        except StopIteration:
            iter = loader.__iter__()  # 終わっていたら最初に戻る
            x = next(iter)
        yield x


class DataSet:
    def __init__(self, data, label):
        self.label = label
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class NegativeSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, source, true_size, sample_negative_size):
        self.true_size = true_size
        self.negative_size = len(source) - true_size
        self.sample_negative_size = sample_negative_size

    def __iter__(self):
        neg = np.arange(self.true_size, self.true_size + self.sample_negative_size)
        indeces = np.concatenate((np.arange(self.true_size), np.random.choice(neg, self.sample_negative_size)))
        np.random.shuffle(indeces)
        for i in indeces:
            yield i

    def __len__(self):
        return self.true_size + self.sample_negative_size


def make_training_dataloader(Training_data_path, train_Ring_num, args, each_nonring_num, NonRing_class):
    ## Training Ring の Dataloader を作成
    Training_Ring_web = (
        webdataset.WebDataset(f"{Training_data_path}/bubble_dataset_train_ring.tar")
        .shuffle(10000000)
        .decode("pil")
        .to_tuple("png", "json")
        .map(preprocess)
    )
    dl_ring_train = torch.utils.data.DataLoader(
        Training_Ring_web,
        collate_fn=od_collate_fn,
        batch_size=args.Ring_mini_batch,
        num_workers=2,
        pin_memory=True,
    )

    ## Training NonRing の Dataloader を作成
    nonring_num = train_Ring_num // len(NonRing_class)
    aug_num = np.delete(np.array(args.NonRing_aug_num) + 1, args.NonRing_remove_class_list)
    NonRing_rsample = np.clip([round(nonring_num / e / a, 5) * 10 for e, a in zip(each_nonring_num, aug_num)], None, 1)
    mini_batch = np.clip(args.NonRing_mini_batch / aug_num / len(NonRing_class), 1, None).astype(int)
    NonRing_web_list = [
        webdataset.WebDataset(f"{Training_data_path}/bubble_dataset_train_nonring_class{cl}.tar")
        .rsample(float(rsample))
        .shuffle(10000000000)
        .decode("pil")
        .to_tuple("png", "json")
        .map(preprocess)
        for rsample, cl in zip(NonRing_rsample, NonRing_class)
    ]
    NonRing_dl_l = [
        torch.utils.data.DataLoader(
            nr_w_l, collate_fn=od_collate_fn, batch_size=int(m_batch), num_workers=2, pin_memory=True
        )
        for nr_w_l, m_batch in zip(NonRing_web_list, mini_batch)
    ]

    return dl_ring_train, [InfiniteIterator(dl) for dl in NonRing_dl_l]  # NonRingを無限にループするイテレータへ


def make_validatoin_dataloader(Validation_data_path, args):
    Dataset_val = (
        webdataset.WebDataset(Validation_data_path)
        .decode("pil")
        .to_tuple("png", "json", "__key__")
        .map(preprocess_validation)
    )
    dl_val = torch.utils.data.DataLoader(
        Dataset_val,
        collate_fn=od_collate_fn_validation,
        batch_size=args.Val_mini_batch,
        num_workers=2,
        pin_memory=True,
    )

    return dl_val
