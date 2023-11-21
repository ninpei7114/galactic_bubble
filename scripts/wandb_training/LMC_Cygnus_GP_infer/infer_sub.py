import copy
import os
import sys

import numpy as np
import torch
import tqdm
from npy_append_array import NpyAppendArray

sys.path.append("../")
from processing import conv, norm_res


def calc_ind(cut_shape, fragment, data_):
    slide_pix = (int(round(cut_shape[0] / fragment)), int(round(cut_shape[1] / fragment)))

    shape = data_.shape
    x_num = int(shape[1] / slide_pix[1]) - 1
    y_num = int(shape[0] / slide_pix[0]) - 1

    x_idx = np.arange(cut_shape[1] / 5, slide_pix[1] * x_num, slide_pix[1])
    y_idx = np.arange(cut_shape[0] / 5, slide_pix[0] * y_num, slide_pix[0])
    x_ind, y_ind = np.meshgrid(x_idx, y_idx)

    l = []
    for x, y in zip(x_ind.ravel(), y_ind.ravel()):
        l.append([y, x])
    ind = np.array(l)
    return ind


def cut_data(data_, many_ind, cut_shape, sig1):
    data_list = []
    position_list_ = []
    for i in many_ind:
        x_min = i[1] - cut_shape / 50
        x_max = i[1] + cut_shape + cut_shape / 50
        y_min = i[0] - cut_shape / 50
        y_max = i[0] + cut_shape + cut_shape / 50
        data_c = data_[int(y_min) : int(y_max), int(x_min) : int(x_max)].view()
        if np.isnan(data_c).any() == False:
            d = copy.deepcopy(data_c)
            d = conv(300, sig1, d)
            d = d[int(cut_shape / 52) : int(cut_shape * 51 / 52), int(cut_shape / 52) : int(cut_shape * 51 / 52)]

            flag = True
            for dim in range(d.shape[2]):
                non_zero_count = np.count_nonzero(d[:, :, dim])
                if non_zero_count >= d.shape[0] * d.shape[1] * 3 / 4:
                    pass
                else:
                    flag = False
            if flag:
                d = norm_res(d)
                data_list.append(d)
                position_list_.append([int(y_min) + int(cut_shape / 50), int(x_min) + int(cut_shape / 50)])
        else:
            pass

    return data_list, position_list_


def infer(ind, batch_size, cut_shape, data_, net_w, detect, args, region, device):
    model_ver = args.model_ver.split("/")[:-2]
    os.makedirs(f"{args.result_save_dir}/{model_ver}/{region}", exist_ok=True)
    sig1 = 1 / (2 * (np.log(2)) ** (1 / 2))
    result_filename = f"{args.result_save_dir}/{model_ver}/{region}/result_ring_select_csize%s.npy" % cut_shape[0]
    position = []
    batch = np.linspace(0, ind.shape[0], batch_size)

    for i in tqdm.tqdm(range(len(batch) - 1)):
        # indを等分して、データを切り取り、推論する
        cut_ind = ind[int(batch[i]) : int(batch[i + 1])]
        data_list, p_list = cut_data(data_, cut_ind, cut_shape[0], sig1)

        if len(data_list) == 0:
            pass
        else:
            p_data = np.array(data_list)
            p_data = p_data.astype(np.float32)
            p_data = torch.from_numpy(p_data)
            pp_data = p_data.permute(0, 3, 1, 2)

            with torch.no_grad():
                net_w.eval()
                pp_data = pp_data.to(device)
                output, _ = net_w(pp_data)
                detections = detect(*output)

                position.append(p_list)
                with NpyAppendArray(result_filename) as npaa:
                    npaa.append(detections.to("cpu").detach().numpy().copy())

    position = np.concatenate(position)
    np.save(f"{args.result_save_dir}/{model_ver}/{region}/position_ring_select_csize%s.npy" % cut_shape[0], position)
