import argparse
import pickle
import time
import math
import os

import faiss
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchinfo import summary

import sys
sys.path.append('../deepcluster/')
import clustering
from util import AverageMeter, Logger, UnifLabelSampler

from utils.ssd_model import SSD
from utils.ssd_model import MultiBoxLoss
from utils.ssd_model import decode

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCluster')

    parser.add_argument('path', metavar='DIR', help='path to dataset')
    parser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 
                        help='device')
    # parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
    #                     choices=['alexnet', 'vgg16'], default='alexnet',
    #                     help='CNN architecture (default: alexnet)')
    # parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    # parser.add_argument('--clustering', type=str, choices=['Kmeans', 'PIC'],
    #                     default='Kmeans', help='clustering algorithm (default: Kmeans)')
    # parser.add_argument('--nmb_cluster', '--k', type=int, default=10,
    #                     help='number of cluster for k-means (default: 10000)')
    # parser.add_argument('--lr', default=0.05, type=float,
    #                     help='learning rate (default: 0.05)')
    # parser.add_argument('--wd', default=-5, type=float,
    #                     help='weight decay pow (default: -5)')
    # parser.add_argument('--reassign', type=float, default=1.,
    #                     help="""how many epochs of training between two consecutive
    #                     reassignments of clusters (default: 1)""")
    # parser.add_argument('--workers', default=4, type=int,
    #                     help='number of data loading workers (default: 4)')
    # parser.add_argument('--epochs', type=int, default=200,
    #                     help='number of total epochs to run (default: 200)')
    # parser.add_argument('--start_epoch', default=0, type=int,
    #                     help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=16, type=int,
                        help='mini-batch size (default: 256)')
    # parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                     help='path to checkpoint (default: None)')
    # parser.add_argument('--checkpoints', type=int, default=25000,
    #                     help='how many iterations between two checkpoints (default: 25000)')
    # parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    # parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    # parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()


class new_vgg(nn.Module):
    def __init__(self, features):
        super(new_vgg, self).__init__()
        
        self.features = features
        self.classifier =  nn.Sequential(
            nn.Linear(512 * 9 * 9, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 3000),
            nn.ReLU()
        )
        self.top_layer = nn.Linear(3000, 2)
        
    def forward(self, x):

        x = self.features(x)
#         x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.top_layer(x)
        
        return x
    
def make_new_model(net):
    pool1 = nn.MaxPool2d(kernel_size=2)
    model = new_vgg(nn.Sequential(*(list(net.vgg)+[pool1])))
#     model = new_vgg(net.vgg)
    return model


def make_ssd_for_deepcluster():
    input_size = 300
    color_mean = (0, 0)
    voc_classes = ['ring']


    ssd_cfg = {
        'num_classes': 2,  # 背景クラスを含めた合計クラス数
        'input_size': input_size,  # 画像の入力サイズ
        'bbox_aspect_num': [4, 4, 4, 4, 4, 4],  # 出力するDBoxのアスペクト比の種類
    #     'bbox_aspect_num': [4, 4, 4, 4, 4, 4],
        'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ   
    #     'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
        'aspect_ratios': [[2], [2], [2], [2], [2], [2]],
    }

    
    net = SSD(phase='train', cfg=ssd_cfg)
    new_model = make_new_model(net)

    return new_model


class DataSet():
    def __init__(self, data):
#         self.label = label
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]#, self.label[index]
    
    
def deepcluster_data(path):
    data = np.load(path)
    data = np.swapaxes(data, 1, 3)
    data = np.swapaxes(data, 2, 3)
    data = data.astype(np.float32)
    
    batch_size = args.batch
    dataset = DataSet(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    return dataloader, dataset


def compute_features(dataloader, model, N):
    """
    compute_featuresは、すべてのデータに対するモデルの出力（データ数, 出力数）のnumpy arrayを作成するパート
    しっかり、cpuに移行している
    """
    # Nは全体のデータ数
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    verbose = False
    if verbose:
        print('Compute features')
#     batch_time = AverageMeter()
    end = time.time()
    # discard the label information in the dataloader
    # 本家は、ImageFolderでデータを呼び出しているため、labelが勝手に着くので、それを捨てている。
    # 今回は、そのようなことをしていない
    # dataloaderのラベル情報を破棄する
    for i, input_tensor in enumerate(dataloader):
        # torch.autograd.Variableで計算グラフの構築を行ってくれる
        # 計算グラフの構築、バックプロぱゲーションに必要な情報の取り扱い
#         input_var = torch.autograd.Variable(input_tensor.to(device))#input_tensor.cuda() #, volatile=True
        input_tensor = input_tensor.to(args.device)
#         print(input_tensor.shape)
        with torch.no_grad():
            model.eval()
            #推論 aux->auxiliary 補助という意味？
            #SSDの場合のauxは、(N, 2)
#             print('aux')
#             aux = model(input_var).to('cpu').detach().numpy()
            aux = model(input_tensor)
            aux = aux.to('cpu').detach().numpy().copy()
#             print('aux : done')
        
        if i == 0:
            # 最初に枠組みの作成を行う
            # featuresのshapeも（N, 2）
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        aux = aux.astype('float32')
        if i < len(dataloader) - 1:
            # featuresに結果を入れていく
            features[i * args.batch: (i + 1) * args.batch] = aux
        else:
            # special treatment for final batch
            # 最後だけはi+1はできないので、このようにしている
            features[i * args.batch:] = aux

            
        end = time.time()

    return features


def train(loader, model, crit, opt, epoch):
    """
        Deep clusterで作成したクラスをラベルとして、学習する

        Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     data_time = AverageMeter()
#     forward_time = AverageMeter()
#     backward_time = AverageMeter()

    # switch to train mode
    model.train()

    # create an optimizer for the last fc layer
    # 最後の全結合層の部分を勾配を出す

    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=1e-3,
        weight_decay=5e-4,
    )

    end = time.time()
    for i, (input_tensor, target) in enumerate(loader):
#         print('input_tensor', input_tensor.shape)
        input_tensor = input_tensor.permute(0, 2, 1, 3).to(args.device)
#         data_time.update(time.time() - end)

        # save checkpoint
        n = len(loader) * epoch + i
        # if n % 25000 == 0:
        #     path = os.path.join(
        #         '',
        #         'checkpoints',
        #         'checkpoint_' + str(n / '') + '.pth.tar',
        #     )
        #     if args.verbose:
        #         print('Save checkpoint at: {0}'.format(path))
        #     torch.save({
        #         'epoch': epoch + 1,
        #         'arch': 'vgg16',
        #         'state_dict': model.state_dict(),
        #         'optimizer' : opt.state_dict()
        #     }, path)

        target = target.to(args.device)
        
        with torch.set_grad_enabled(True):
            model.train()

            output = model(input_tensor)
            loss = crit(output, target)

            # compute gradient and do SGD step
            opt.zero_grad()
            optimizer_tl.zero_grad()

            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
#             print('loss : done')
            opt.step()
            optimizer_tl.step()

        # measure elapsed time
#         batch_time.update(time.time() - end)
#         end = time.time()

#         if (i % 200) == 0:
#             print('Epoch: [{0}][{1}/{2}]\t'
#                   'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss: {loss.val:.4f} ({loss.avg:.4f})'
#                   .format(epoch, i, len(loader), batch_time=batch_time,
#                           data_time=data_time, loss=losses))

    return 0 #losses.avg


def main(args):
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = make_ssd_for_deepcluster()
    fd = int(model.top_layer.weight.size()[1])
    model.to(args.device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3, momentum=0.9,
        weight_decay=5e-4,
    )
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    deepcluster = clustering.Kmeans(2)

    dataloader, dataset = deepcluster_data(args.path)

    for epoch in range(1, 10):
    #     model.top_layer = None ## vggの全結合層部分を消している？
    #     model_ = model
        torch.manual_seed(31)
        torch.cuda.manual_seed_all(31)
        np.random.seed(31)

        model.classifier = nn.Sequential(*list(model.classifier.children()))
        features = compute_features(dataloader, model, len(dataset))

        clustering_loss = deepcluster.cluster(features, verbose=True)
        reassign = 1
        workers = 4

        train_dataset = clustering.cluster_assign(deepcluster.images_lists,
                                                      dataset.data)

        # uniformly sample per target
        sampler = UnifLabelSampler(int(reassign * len(train_dataset)),
                                   deepcluster.images_lists)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch,
            num_workers=workers,
            sampler=sampler,
            pin_memory=True,
        )

        print('train : start')
        loss = train(train_dataloader, model, criterion, optimizer, epoch)
        print('train : done')
        torch.save({'epoch': epoch + 1,
                        'arch':  'vgg16',
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict()},
                       os.path.join('', 'checkpoint.pth.tar'))

    #     cluster_log.log(deepcluster.images_lists)



if __name__ == '__main__':
    args = parse_args()
    main(args)
