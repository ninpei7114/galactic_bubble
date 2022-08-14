import numpy as np
import pandas as pd
import ast
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='make data for deepcluster')

    parser.add_argument('ring_path', metavar='DIR', help='path to ring')
    parser.add_argument('train_non_ring_path', metavar='DIR', help='path to train non_ring')
    parser.add_argument('val_non_ring_path', metavar='DIR', help='path to val non_ring')
    parser.add_argument('train_Non_Ring_num', type=int, help='train Non_Ring number')
    parser.add_argument('val_Non_Ring_num', type=int, help='val Non_Ring number')
    parser.add_argument('savedir', default='.', 
                        help='data save dir')


    return parser.parse_args()


def main(args):

    train_label = pd.read_csv(args.ring_path+'/ring/train_ring_label.csv')
    train_data = np.load(args.ring_path+'/ring/train_ring_data.npy')
    val_label = pd.read_csv(args.ring_path+'/ring/val_ring_label.csv')
    val_data = np.load(args.ring_path+'/ring/val_ring_data.npy')


    train_label = train_label.drop('Unnamed: 0', axis=1)
    val_label = val_label.drop('Unnamed: 0', axis=1)

    train_label['xmin'] = [ast.literal_eval(d) for d in train_label['xmin']]
    train_label['xmax'] = [ast.literal_eval(d) for d in train_label['xmax']]
    train_label['ymin'] = [ast.literal_eval(d) for d in train_label['ymin']]
    train_label['ymax'] = [ast.literal_eval(d) for d in train_label['ymax']]

    val_label['xmin'] = [ast.literal_eval(d) for d in val_label['xmin']]
    val_label['xmax'] = [ast.literal_eval(d) for d in val_label['xmax']]
    val_label['ymin'] = [ast.literal_eval(d) for d in val_label['ymin']]
    val_label['ymax'] = [ast.literal_eval(d) for d in val_label['ymax']]


    no_ring_train_ = np.load(args.train_non_ring_path)
    no_ring_val_ = np.load(args.val_non_ring_path)


    train_data = np.concatenate([train_data, no_ring_train_[np.random.randint(0, no_ring_train_.shape[0], int(args.train_Non_Ring_num))]])
    val_data = np.concatenate([val_data, no_ring_val_[np.random.randint(0, no_ring_val_.shape[0], int(args.val_Non_Ring_num))]])


    for i in range(args.train_Non_Ring_num):
        
        train_label = pd.concat([train_label, pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax', 'id'], 
                data= [[[] for i in range(7)]])])


    for i in range(args.val_Non_Ring_num):
        val_label = pd.concat([val_label, pd.DataFrame(columns=['fits', 'name', 'xmin', 'xmax', 'ymin', 'ymax', 'id'], 
                data= [[[] for i in range(7)]])])


    print(train_data.shape, len(train_label), val_data.shape, len(val_label))




    np.save(args.savedir+'/train.npy', train_data)
    np.save(args.savedir+'/val.npy', val_data)


    train_label.to_csv(args.savedir+'/train_label.csv')
    val_label.to_csv(args.savedir+'/val_label.csv')



if __name__ == '__main__':
    args = parse_args()
    main(args)

    