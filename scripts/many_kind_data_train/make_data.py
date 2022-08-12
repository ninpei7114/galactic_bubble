# from make_NonRing  import make_nonring
from make_Ring_data import make_ring

import numpy as np
import pandas as pd


def make_data(spitzer_path, name, pattern, f_log):

    train_data, train_label, val_data, val_label = make_ring(spitzer_path, name, pattern)
    train_Ring_num, val_Ring_num = train_data.shape[0], val_data.shape[0]

    no_Ring_train = np.load('NonRing/no_ring_300_9000_train.npy')
    no_Ring_val = np.load('NonRing/no_ring_300_900_val.npy')
    no_Ring_train_moyamoya = np.load('NonRing/no_ring_moyamoya_train.npy')
    no_Ring_val_moyamoya = np.load('NonRing/no_ring_moyamoya_val.npy')

    no_Ring_train_random = np.random.randint(0, no_Ring_train.shape[0], int(train_data.shape[0]))
    no_Ring_train_moyamoya_random = np.random.randint(0, no_Ring_train_moyamoya.shape[0], int(train_data.shape[0]))
    no_Ring_val_random = np.random.randint(0, no_Ring_val.shape[0], int(val_data.shape[0]/2))
    no_Ring_val_moyamoya_random = np.random.randint(0, no_Ring_val_moyamoya.shape[0], int(val_data.shape[0]/2))

    np.save(name+'/no_Ring_train_random.npy', no_Ring_train_random)
    np.save(name+'/no_Ring_train_moyamoya_random.npy', no_Ring_train_moyamoya_random)
    np.save(name+'/no_Ring_val_random.npy', no_Ring_val_random)
    np.save(name+'/no_Ring_val_moyamoya_random.npy', no_Ring_val_moyamoya_random)

    train_data = np.concatenate([train_data, no_Ring_train[no_Ring_train_random], 
                             no_Ring_train_moyamoya[no_Ring_train_moyamoya_random]])

    val_data = np.concatenate([val_data, no_Ring_val[no_Ring_val_random], 
                             no_Ring_val_moyamoya[no_Ring_val_moyamoya_random]])



    for _ in range(int(train_Ring_num)*2):
        train_label = pd.concat([train_label, pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax', 'id'], 
             data= [[[] for i in range(7)]])])

    for _ in range(int(val_Ring_num/2)*2):
        val_label = pd.concat([val_label, pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax', 'id'], 
             data= [[[] for i in range(7)]])])


    print('train data shape : ', train_data.shape)
    print('train label length : ', len(train_label))
    print('val data shape : ', val_data.shape)
    print('val label length : ', len(val_label))

    f_log.write('====================================\n')
    f_log.write('Train Ring num  : %s .\n'%train_Ring_num)
    f_log.write('Val Ring num  : %s .\n'%val_Ring_num)
    f_log.write('> \n')
    f_log.write('Total Train num  : %s .\n'%str(train_data.shape))
    f_log.write('Total Val num  : %s .\n'%str(val_data.shape))
    f_log.write('> \n')
    f_log.write('Total Train label length  : %s .\n'%len(train_label))
    f_log.write('Total Val label length  : %s .\n'%len(val_label))
    f_log.write('> \n')


    train_label = train_label.reset_index()
    val_label = val_label.reset_index()

    train_label_list = []
    for i in range(len(train_label)):
        lab = []
        for k in range(len(train_label['xmin'][i])):
            labe = []
            labe.append(train_label['xmin'][i][k])
            labe.append(train_label['ymin'][i][k])
            labe.append(train_label['xmax'][i][k])
            labe.append(train_label['ymax'][i][k])
            labe.append(0)
            lab.append(labe)
        train_label_list.append(np.array(lab))

    val_label_list = []
    for i in range(len(val_label)):
        lab = []
        for k in range(len(val_label['xmin'][i])):
            labe = []
            labe.append(val_label['xmin'][i][k])
            labe.append(val_label['ymin'][i][k])
            labe.append(val_label['xmax'][i][k])
            labe.append(val_label['ymax'][i][k])
            labe.append(0)
            lab.append(labe)
        val_label_list.append(np.array(lab))

    train_data = train_data[:,:,:,:2]
    train_data = np.swapaxes(train_data, 2, 3)
    train_data = np.swapaxes(train_data, 1, 2)

    val_data = val_data[:,:,:,:2]
    val_data = np.swapaxes(val_data, 2, 3)
    val_data = np.swapaxes(val_data, 1, 2)

    return train_data, train_label_list, val_data, val_label_list, train_Ring_num, val_Ring_num

