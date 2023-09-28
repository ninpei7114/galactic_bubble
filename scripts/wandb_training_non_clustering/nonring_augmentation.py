import numpy as np
import torch
from skimage import transform


def augmentation_func(data, augmentation_type):
    data_l = []

    for i in data:
        if augmentation_type == 0:
            data_l.append(np.flipud(i))

        elif augmentation_type == 1:
            data_l.append(np.fliplr(i))

        elif augmentation_type == 2:
            data_l.append(transform.rotate(i, 90))

        elif augmentation_type == 3:
            data_l.append(transform.rotate(i, 180))

        elif augmentation_type == 4:
            data_l.append(transform.rotate(i, 270))

    return np.array(data_l)


def nonring_augmentation(iter_noring_list, NonRing_class_num, NonRing_rg, args):
    non_ring_image = []
    non_ring_label = []

    for class_num, noring in zip(NonRing_class_num, iter_noring_list):
        data, label = next(noring)
        non_ring_image.append(data)
        non_ring_label.extend(label)
        NonRing_aug_num = args.NonRing_aug_num[class_num]

        if NonRing_aug_num == 0:
            pass
        else:
            for flag in NonRing_rg.choice(np.arange(5), NonRing_aug_num, replace=False):
                non_ring_image.append(augmentation_func(data, flag))
                non_ring_label.extend([torch.tensor([]) for i in range(data.shape[0])])

    return np.concatenate(non_ring_image), non_ring_label
