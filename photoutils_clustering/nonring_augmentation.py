import numpy as np
import torch
from skimage import transform


def augmentation_func(data, augmentation_type):
    if augmentation_type == 1:
        return np.flipud(data)
    elif augmentation_type == 2:
        return np.fliplr(data)
    elif augmentation_type == 3:
        return transform.rotate(data, 90)
    elif augmentation_type == 4:
        return transform.rotate(data, 180)
    elif augmentation_type == 5:
        return transform.rotate(data, 270)
    elif augmentation_type == 6:
        return transform.rotate(np.fliplr(data), 90)
    elif augmentation_type == 7:
        return transform.rotate(np.fliplr(data), 180)
    elif augmentation_type == 8:
        return transform.rotate(np.fliplr(data), 270)
    elif augmentation_type == 9:
        return transform.rotate(np.flipud(data), 90)
    elif augmentation_type == 10:
        return transform.rotate(np.flipud(data), 180)
    elif augmentation_type == 11:
        return transform.rotate(np.flipud(data), 270)
    elif augmentation_type == 12:
        return data


def nonring_augmentation(iter_noring_list, NonRing_class_num, NonRing_rg, args):
    non_ring_image = []
    non_ring_label = []

    for class_num, noring in zip(NonRing_class_num, iter_noring_list):
        data, label = next(noring)
        NonRing_aug_num = args.NonRing_aug_num[class_num]

        if NonRing_aug_num == 0:
            non_ring_image.append(data)
            non_ring_label.extend(label)
        else:
            for i in data:
                flag = NonRing_rg.choice(np.arange(12), 1, replace=False)
                if flag == 0:
                    non_ring_image.append(i[None])
                    non_ring_label.append(torch.tensor([]))
                else:
                    non_ring_image.append(augmentation_func(i, flag)[None])
                    non_ring_label.append(torch.tensor([]))

    return np.concatenate(non_ring_image), non_ring_label
