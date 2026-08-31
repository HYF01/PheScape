import os
from typing import Optional
from anndata import AnnData
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import issparse, csr_matrix
import torch
from pathlib import Path

os.environ['R_HOME'] = 'E:\\R\\R-4.2.2'
os.environ['R_USER'] = 'E:\\environment2023\\Anaconda3\\envs\\ae38\\Lib\\site-packages\\rpy2'


def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:
    """One hot a tensor of categories."""
    index = index.reshape((-1, 1))
    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='OURS', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def spatial_reconstruction(
        coord,
        adata: AnnData,
        alpha: float = 1,
        n_neighbors: int = 10,
        n_pcs: int = 15,
        use_highly_variable: Optional[bool] = False,
        normalize_total: bool = False,
        copy: bool = True,

) -> Optional[AnnData]:
    adata = adata.copy() if copy else adata
    adata.layers['counts'] = adata.X

    adata.layers['log1p'] = adata.X

    sc.pp.pca(adata, n_comps=n_pcs, use_highly_variable=use_highly_variable)

    # coord = adata.obsm['spatial']
    neigh = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(coord)
    nbrs = neigh.kneighbors_graph(coord)

    dists = np.exp(2 - cosine_distances(adata.obsm['X_pca'])) - 1
    conns = nbrs.T.toarray() * dists

    X = adata.X.toarray() if issparse(adata.X) else adata.X
    X_rec = alpha * np.matmul(conns / np.sum(conns, axis=0, keepdims=True), X) + X

    adata.X = csr_matrix(X_rec)

    adata.layers['log1p-aug'] = adata.X  # 把增强矩阵更新到data.X里面，log1p也更新
    '''增强表达矩阵标准化、归一化、取2000高变化基因  '''
    exmatrix_aug = adata.to_df(layer='log1p-aug').to_numpy()

    del adata.obsm['X_pca']

    adata.uns['spatial_reconstruction'] = {}

    rec_dict = adata.uns['spatial_reconstruction']

    rec_dict['params'] = {}
    rec_dict['params']['alpha'] = alpha
    rec_dict['params']['n_neighbors'] = n_neighbors
    rec_dict['params']['n_pcs'] = n_pcs
    rec_dict['params']['use_highly_variable'] = use_highly_variable
    rec_dict['params']['normalize_total'] = normalize_total

    return adata


def Z_leiden(adata: AnnData,
             numcluster: int,
             Z,
             start_resol,
             annotation_name):
    adata.uns['Z'] = dict()
    neighbors_dict = adata.uns['Z']
    adata.obsp['Z'] = Z
    neighbors_dict['connectivities_key'] = 'Z'
    neighbors_dict['distances_key'] = 'Z'
    print(np.sum(Z > 0) / (Z.shape[0] * Z.shape[1]))

    '''自适应调节leiden分辨率'''
    ari_dict = {}
    resolution = start_resol  # 0.9
    sc.tl.leiden(adata, resolution=resolution, neighbors_key='Z')
    print(len(set(adata.obs["leiden"].tolist())))
    while len(set(adata.obs["leiden"].tolist())) <= numcluster:
        resolution = resolution + 0.001  # 0.015
        sc.tl.leiden(adata, resolution=resolution, neighbors_key='Z')
        if len(set(adata.obs["leiden"])) == numcluster:
            df_ari = pd.DataFrame({'label': adata.obs[annotation_name], 'leiden': adata.obs['leiden']})
            ARI = adjusted_rand_score(df_ari['label'], df_ari['leiden'])
            ari_dict[resolution] = ARI
        else:
            pass
        print("resolution={},num of clusters={}".format(resolution, len(set(adata.obs["leiden"].tolist()))))
    best_resolu = max(ari_dict, key=ari_dict.get)
    sc.tl.leiden(adata, resolution=best_resolu, neighbors_key='Z')
    print("最佳聚类分辨率为：{}, ARI={}".format(best_resolu, ari_dict[best_resolu]))

    # fig, axs = plt.subplots(figsize=(10, 8.25))
    # sc.pl.spatial(
    #     adata,
    #     img_key='hires',
    #     color='leiden',
    #     size=1.5,
    #     color_map='Reds',
    #     legend_loc='right margin',
    #     frameon=False,
    #     title='leiden ARI={}'.format(ARI),
    #     # vmin=0,
    #     # vmax=1,
    #     show=True,
    #     ax=axs,
    # )
    #
    # plt.tight_layout()
    return ari_dict[best_resolu]


def qz_mclust(adata: AnnData,
              numcluster: int,
              qz,
              annotation_name,
              sample
              ):
    '''把20维的embedding写入adata.obsm['OURS']'''
    if sample == 'CT_2-5':
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
        cells = np.array(X.index)
        cell_reps = pd.DataFrame(qz)
        cell_reps.index = cells
        adata.obsm['OURS'] = cell_reps.loc[adata.obs_names,].values
        '''调用pp.neighbor,和mcluster'''
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='OURS')
        adata = mclust_R(adata, used_obsm='OURS', num_cluster=numcluster)
        '''合并几个相同clust'''
        df_mclust = pd.DataFrame(index=adata.obs.index)
        df_mclust['first'] = adata.obs['mclust']
        df_mclust = df_mclust.astype(float)
        print(set(df_mclust['first']))

        df_mclust = df_mclust.replace(5, 7)
        df_mclust = df_mclust.replace(2, 5)
        df_mclust = df_mclust.replace(6, 7)
        df_mclust = df_mclust.replace(3, 6)
        df_mclust = df_mclust.replace(4, 6)
        print(set(df_mclust['first']))

        '''把clust1进一步划分成4份'''
        adata_1 = adata[adata.obs['mclust'] == 1, :]
        sc.pp.neighbors(adata_1, n_neighbors=4, use_rep='OURS')
        adata_1 = mclust_R(adata_1, used_obsm='OURS', num_cluster=4)
        df_1 = pd.DataFrame(index=adata_1.obs.index)
        df_1['second'] = adata_1.obs['mclust']
        df_1 = df_1.astype(float)
        print(set(df_1['second']))

        df_all = df_mclust.merge(df_1, left_index=True, right_index=True, how='outer')
        print(df_all)
        for i in range(df_all.shape[0]):
            if pd.isnull(df_all.iloc[i, 1]):
                pass
            else:
                df_all.iloc[i, 0] = df_all.iloc[i, 1]
        print(df_all)
        print(set(df_all['first']))

        adata.obs['mclust'] = df_all['first']
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
        print(adata.obs['mclust'])

    else:
        X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
        cells = np.array(X.index)
        cell_reps = pd.DataFrame(qz)
        cell_reps.index = cells
        adata.obsm['OURS'] = cell_reps.loc[adata.obs_names,].values
        '''调用pp.neighbor,和mcluster'''
        sc.pp.neighbors(adata, n_neighbors=10, use_rep='OURS')
        adata = mclust_R(adata, used_obsm='OURS', num_cluster=numcluster)

    df_ari = pd.DataFrame({'label': adata.obs[annotation_name], 'mclust': adata.obs['mclust']})
    ARI = adjusted_rand_score(df_ari['label'], df_ari['mclust'])
    return ARI


def domain_plot(adata: AnnData,
                sample,
                my_palette,
                numcluster: int,
                wide,
                length,
                spot_size,
                Z,
                qz,
                annotation_name,
                start_resol=0.9,
                ):
    # ARI_leiden = Z_leiden(adata, numcluster, Z, start_resol, annotation_name)
    ARI_mclust = qz_mclust(adata, numcluster, qz, annotation_name)

    fig, axs = plt.subplots(figsize=(wide, length))
    sc.pl.embedding(
        adata,
        basis='spatial',
        palette=my_palette,
        colorbar_loc=None,
        frameon=False,
        legend_loc='none',  # 'right margin',
        size=spot_size,
        title='ours ARI={}'.format(ARI_mclust),
        color='mclust',  # batch
        show=False,
        ax=axs  # y轴反向,,
    )
    plt.savefig("{}_mclust_domain.png".format(sample), dpi=900)

    sc.pl.spatial(
        adata,
        img_key='hires',
        color=['mclust'],
        size=1.5,
        color_map='Reds',
        palette=my_palette,
        legend_loc='right margin',
        frameon=False,
        title=['mclust ARI={}'.format(ARI_mclust)],
        # vmin=0,
        # vmax=1,
        show=False,
        # ax=axs,
    )
    # plt.tight_layout()
    # plt.savefig("{}_mclust_domain.png".format(sample), dpi=900)

    #
    # color_10 = ['#5b8ff9', '#5ad8a6', '#5d7092', '#f6bd16', '#e8684a',
    #             '#DB4C6C', '#9270ca', '#FFE4B5', '#269a99', '#ff99ca']
    # ax = sc.pl.scatter(adata,
    #                    alpha=1,
    #                    x="y_pixel",
    #                    y="x_pixel",
    #                    color='mclust',
    #                    title='mclust ARI={}'.format(ARI_mclust),
    #                    color_map='Reds',
    #                    palette=my_palette,
    #                    show=True,
    #                    size=58)
