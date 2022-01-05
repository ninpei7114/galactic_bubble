# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']


def pil_loader(path):
    """Loads an image.
    Args:
        path (string): path to image file
    Returns:
        Image
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ReassignedDataset(data.Dataset):
    """A dataset where the new images labels are given in argument.
    Args:
        image_indexes (list): list of data indexes
        pseudolabels (list): list of labels for each data
        dataset (list): list of tuples with paths to images
        transform (callable, optional): a function/transform that takes in
                                        an PIL image and returns a
                                        transformed version
    """

    def __init__(self, image_indexes, pseudolabels, dataset, transform=None):
        self.imgs = self.make_dataset(image_indexes, pseudolabels, dataset)
        self.transform = transform

    def make_dataset(self, image_indexes, pseudolabels, dataset):
        label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
        images = []
        for j, idx in enumerate(image_indexes):
            # path = dataset[idx][0]
            img = dataset[idx]
            # print(img.shape)
            # print(path)
            pseudolabel = label_to_idx[pseudolabels[j]]
            # images.append((path, pseudolabel))
            images.append((img, pseudolabel))
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): index of data
        Returns:
            tuple: (image, pseudolabel) where pseudolabel is the cluster of index datapoint
        """
        # path, pseudolabel = self.imgs[index]
        img, pseudolabel = self.imgs[index]
        # print('__getitem__shape', img.shape)
        # print(path)
        # img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pseudolabel

    def __len__(self):
        return len(self.imgs)


def preprocess_features(npdata, pca=2):
    """
    features、つまりモデルの出力(npdata (np.array N * ndim))を

    Preprocess an array of features.
    Args:
        
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # faissで主成分分析
    # ndimをpca＝2次元に次元削減
    # 主成分分析とは次元を削減できるもの、分散によって次元の重要度を決める
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=0)
    # 主成分分析するためのパラメーターの設定
    mat.train(npdata)
    # print('mat.train(npdata) : done')
    assert mat.is_trained
    # npdataを当てはめている
    # print(npdata)
    npdata = mat.apply_py(npdata)
    # print(npdata)
    # L2 normalizationをしている
    # row_sumsは、１次元目の値を２乗した和の合計
    row_sums = np.linalg.norm(npdata, axis=1)
    # print(row_sums)
    # row_sums[:, np.newaxis]で二次元に
    # 
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """
    xbは、モデルの出力を主成分分析したもの

    Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape
    print(xb.shape)

    # we need only a StandardGpuResources per GPU
    # print('faiss.StandardGpuResources : start')
    # res = faiss.StandardGpuResources()
    # print('faiss.StandardGpuResources : done')

    # # L2
    # print('faiss.GpuIndexFlatConfig : start')
    # flat_config = faiss.GpuIndexFlatConfig()
    # print('faiss.GpuIndexFlatConfig : done')
    # flat_config.device = int(torch.cuda.device_count()) - 1
    # print(xb.shape)
    ## ここでエラー
    # print('faiss.GpuIndexFlatL2 : start')
    # index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index = faiss.IndexFlatL2(dim)
    # print('faiss.GpuIndexFlatL2 : done')
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


def cluster_assign(images_lists, dataset):
    """
    
    Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])
#     t = transforms.Compose([transforms.RandomResizedCrop(224),
#                             transforms.RandomHorizontalFlip(),
#                             transforms.ToTensor(),
#                             normalize])
    t = transforms.Compose([transforms.ToTensor(),
                            ])

    return ReassignedDataset(image_indexes, pseudolabels, dataset, t)


def run_kmeans(x, nmb_clusters, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape
    
    # print('x.shape', x.shape)
    # print(nmb_clusters)
    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000

    # res = faiss.StandardGpuResources()

    # flat_config.useFloat16 = False
    # flat_config.device = 0
    # index = faiss.GpuIndexFlatL2(res, d, flat_config)
    index = faiss.IndexFlatL2(d)
    # print(np.sum(x))
    # print(np.sum(index))
    clus.train(x, index)
    _, I = index.search(x, 1)

    # numpyのarrayに変換しているだけ
    losses = faiss.vector_to_array(clus.centroids)

    # losses = 213
    # if verbose:
    #     print('k-means loss evolution: {0}'.format(losses))
    # print()
#     return [int(n[0]) for n in I], losses[-1]
    return [int(n[0]) for n in I]#, losses


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def cluster(self, data, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data)

        # cluster the data
        # print('run_kmeans : start')
#         I, loss = run_kmeans(xb, self.k, verbose)
        I = run_kmeans(xb, self.k, verbose)
        loss = 0
        # print('run_kmeans : done')
        self.images_lists = [[] for i in range(self.k)]
        # print('self.k : ', self.k)
        # print('len(data) : ', len(data))
        # print('len(I) : ', len(I))
        for i in range(len(data)):
            # try:
            self.images_lists[I[i]].append(i)
            # except:
            #     print(I[i])

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC(object):
    """
    最近傍グラフに対して、PICを行う
    PICは、データ間の類似度をもとにクラスタリングする
    Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons
        
        
    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        # モデルの出力を主成分分析して、次元削減している
        xb = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0
