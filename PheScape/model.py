from typing import Optional
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from anndata import AnnData
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.python.platform import self_check
from tqdm import trange
import scanpy as sc
# from module import ContrastiveVAE_SC,ContrastiveVAE_BK
from utils import spatial_reconstruction, one_hot, Z_leiden, qz_mclust, domain_plot
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from typing import Callable, Optional
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import pandas as pd


def data_read(dir_path: str):
    with open(dir_path, 'r') as f:
        raw_data = f.readline()
        data = raw_data[1:-1].split(',')
    return np.asfarray(data, float)

def data_conbine(
        data1: AnnData,
        data2: AnnData,
        device: Optional[str] = 'cuda'
    ):
    adata_st_sc = sc.AnnData.concatenate(data1, data2)  # 两个数据直接拼接，样本相加，基因取交集，关键添加了batch槽
    batch_info = pd.Categorical(adata_st_sc.obs['batch'])  # 每个样本（包括空转和cell）的batch号码 ['0','0',...'1']Length: 15327
    n_batch = batch_info.categories.shape[0]  # batch的种类数目  Categories (2, object): ['0', '1']
    batch_index = batch_info.codes.copy()  # 分类变量转换为int数字编码，返回Series对象  [0 0 0 ... 1 1 1]
    batch_index = one_hot(torch.Tensor(batch_index).to(device),
                          n_batch).cpu().numpy()  # 将数字编码转换为ont-hot编码[[1. 0.] [1. 0.] [0. 1.] [0. 1.]]
    adata_st_sc.obsm['batch_index'] = batch_index  # adata0添加槽 obsm: 'batch_index'
    data1_new = adata_st_sc[adata_st_sc.obs['batch'] == '0', :]  # 加完batch后重新划分单细胞、空转，obsm: 'batch_index'也会按照样本重新划分
    data2_new = adata_st_sc[adata_st_sc.obs['batch'] == '1', :]
    return data1_new,data2_new



class Layer1(nn.Module):
    def __init__(
            self,
            n_in: int,
            n_out: int,
            dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.network(x)


class Layer2(nn.Module):

    def __init__(
            self,
            n_in: int = 128,
            n_out: int = 10,
            var_eps: float = 1e-4,
            var_activation: Optional[Callable] = None,
    ):
        super().__init__()

        self.var_eps = var_eps
        self.mean_encoder = nn.Linear(n_in, n_out)
        self.var_encoder = nn.Linear(n_in, n_out)
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x):
        q_m = self.mean_encoder(x)
        q_v = self.var_activation(self.var_encoder(x)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()

        return dist, latent


class Layer3(nn.Module):

    def __init__(
            self,
            n_in: int,
            n_out: int,
            dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.network(x)


class Layer4(nn.Module):

    def __init__(
            self,
            n_in: int,
            n_out: int,
            dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class LayerZ(nn.Module):

    def __init__(
            self,
            n_obs: int,
    ):
        super().__init__()
        self.Z = torch.nn.Parameter(1.0e-8 * torch.ones((n_obs, n_obs)))

    def forward(self, x):
        torch.diagonal(self.Z.data).fill_(0)
        self.Z.data = torch.abs(nn.LeakyReLU()(self.Z))
        self.Z.data[self.Z < 1e-5] = 0
        return torch.matmul(self.Z, x)

    def getZ(self):
        torch.diagonal(self.Z.data).fill_(0)
        self.Z.data = torch.abs(nn.LeakyReLU()(self.Z))
        self.Z.data[self.Z < 1e-5] = 0
        return self.Z.detach().cpu().numpy()


class LayerS_ST_CL(nn.Module):

    def __init__(
            self,
            n_ST: int,  # the number of spots (ST)
            n_CL: int,  # the number of Cell line
    ):
        super().__init__()
        self.S = torch.nn.Parameter(1.0e-8 * torch.ones((n_ST, n_CL)))

    def forward(self, x):
        self.S.data = torch.abs(nn.LeakyReLU()(self.S))
        self.S.data[self.S < 1e-5] = 0
        return torch.matmul(self.S, x)

    def getS(self):
        self.S.data = torch.abs(nn.LeakyReLU()(self.S))
        self.S.data[self.S < 1e-5] = 0
        return self.S.detach().cpu().numpy()

class LayerS_ST_BK(nn.Module):

    def __init__(
            self,
            n_ST: int,  # the number of spots (ST)
            n_BK: int,  # the number of Cell line
    ):
        super().__init__()
        self.S = torch.nn.Parameter(1.0e-8 * torch.ones((n_ST, n_BK)))

    def forward(self, x):
        self.S.data = torch.abs(nn.LeakyReLU()(self.S))
        self.S.data[self.S < 1e-5] = 0
        return torch.matmul(self.S, x)

    def getS(self):
        self.S.data = torch.abs(nn.LeakyReLU()(self.S))
        self.S.data[self.S < 1e-5] = 0
        return self.S.detach().cpu().numpy()



class LayerV(nn.Module):

    def __init__(
            self,
            n_obs: int,
            ref_idx,
    ):
        super().__init__()

        self.ref_idx = ref_idx
        self.n_ref = ref_idx.shape[0]
        self.V = torch.nn.Parameter(1.0e-8 * torch.ones((n_obs, self.n_ref)))

    def forward(self, x, idx):
        self.V.data[self.ref_idx, range(self.n_ref)] = 0
        return torch.matmul(nn.ReLU()(torch.index_select(self.V, dim=0, index=idx)), x)

    def getV(self):
        self.V.data[self.ref_idx, range(self.n_ref)] = 0
        return nn.ReLU()(self.V).detach().cpu().numpy()


class map1(nn.Module):

    def __init__(
            self,
            n_covar: int,
            n_input: int,  # the number of genes (ST) b=d
            n_obs: int,  # the number of spots a
            n_cl: int,  # the number of cell line c
            n_hidden: int,
            n_latent: int,
            dropout_rate: float = 0.1,
            var_activation: Optional[Callable] = None,

    ):
        super().__init__()

        self.n_covar = n_covar

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        self.layer1 = Layer1(
            n_in=n_input,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.layer2 = Layer2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )

        self.layer3 = Layer3(
            n_in=n_latent + n_covar,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.layer4 = Layer4(
            n_in=n_hidden,
            n_out=n_input,
        )

        self.layerZ = LayerZ(
            n_obs=n_obs,
        )

        self.layerS_ST_CL = LayerS_ST_CL(
            n_ST=n_obs,
            n_CL=n_cl,
        )


    def inference(self, x):

        x1 = self.layer1(x)
        qz, z = self.layer2(x1)

        return dict(x1=x1, z=z, qz=qz)

    def generative(self, z, covar):

        if covar is None:
            x3 = self.layer3(z)
        else:
            x3 = self.layer3(torch.cat((z, covar), dim=-1))

        x4 = self.layer4(x3)

        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(x3=x3, x4=x4, pz=pz)

    def loss(
            self,
            x_st,  # 空转矩阵
            x_cl,  # cell line矩阵
            inference_outputs_ST,  # 空转：dict(hidden:x1,qz,z)
            generative_outputs_ST,  # 空转：输出 dict(hidden-x3,output-x4,pz)
            inference_outputs_CL,  # cell line：dict(hidden:x1,qz,z)
            generative_outputs_CL,  # cell line：dict(hidden-x3,output-x4,pz)
            Z_weight,  # epoch / n_epochs
    ):
        kl_divergence_z_ST = kl(inference_outputs_ST['qz'], generative_outputs_ST['pz']).sum(dim=1)
        kl_divergence_z_CL = kl(inference_outputs_CL['qz'], generative_outputs_CL['pz']).sum(dim=1)

        recon_loss_ST = torch.norm(x_st - generative_outputs_ST['x4'])  # 空转重构损失
        recon_loss_CL = torch.norm(self.layerS_ST_CL(x_cl) - generative_outputs_CL['x4'])  # cell line重构损失


        constrastive_loss = (torch.norm(inference_outputs_ST['x1'] - inference_outputs_CL['x1']) +  # x1的对比损失
                              torch.norm(inference_outputs_ST['z'] - inference_outputs_CL['z']) +  # laten的对比损失
                              torch.norm(generative_outputs_ST['x3'] - generative_outputs_CL['x3']))# x3的对比损失

        loss = ((15.0 * (torch.mean(kl_divergence_z_ST) + recon_loss_ST)
                + 0.1 * (torch.mean(kl_divergence_z_CL) + recon_loss_CL))
                + 1.5 * constrastive_loss
                )

        # decon 15 0.1 1.0
        # sim 15 0.1 1.0 1.5

        if Z_weight > 0.5:
            loss += Z_weight * torch.norm(inference_outputs_ST['x1'] - self.layerZ(inference_outputs_ST['x1']))
            loss += Z_weight * torch.norm(inference_outputs_ST['z'] - self.layerZ(inference_outputs_ST['z']))
            loss += Z_weight * torch.norm(generative_outputs_ST['x3'] - self.layerZ(generative_outputs_ST['x3']))

            loss += Z_weight * torch.norm(inference_outputs_CL['x1'] - self.layerZ(inference_outputs_CL['x1']))
            loss += Z_weight * torch.norm(inference_outputs_CL['z'] - self.layerZ(inference_outputs_CL['z']))
            loss += Z_weight * torch.norm(generative_outputs_CL['x3'] - self.layerZ(generative_outputs_CL['x3']))
        return loss



class map2(nn.Module):

    def __init__(
            self,
            n_covar: int,
            n_input: int,  # the number of genes (ST) b=d
            n_obs: int,  # the number of spots a
            n_bk: int,  # the number of bulk sample
            n_hidden: int,
            n_latent: int,
            dropout_rate: float = 0.1,
            var_activation: Optional[Callable] = None,

    ):
        super().__init__()

        self.n_covar = n_covar

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        self.layer1 = Layer1(
            n_in=n_input,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.layer2 = Layer2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )

        self.layer3 = Layer3(
            n_in=n_latent + n_covar,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.layer4 = Layer4(
            n_in=n_hidden,
            n_out=n_input,
        )

        self.layerZ = LayerZ(
            n_obs=n_obs,
        )


        self.layerS_ST_BK = LayerS_ST_BK(
            n_ST=n_obs,
            n_BK=n_bk,
        )

    def inference(self, x):

        x1 = self.layer1(x)
        qz, z = self.layer2(x1)

        return dict(x1=x1, z=z, qz=qz)

    def generative(self, z, covar):
        if covar is None:
            x3 = self.layer3(z)
        else:
            x3 = self.layer3(torch.cat((z, covar), dim=-1))

        x4 = self.layer4(x3)

        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(x3=x3, x4=x4, pz=pz)

    def loss(
            self,
            x_st,  # 空转矩阵
            x_bk,  # bulk 矩阵
            inference_outputs_ST,  # 空转：dict(hidden:x1,qz,z)
            generative_outputs_ST,  # 空转：输出 dict(hidden-x3,output-x4,pz)
            inference_outputs_BK,  # BULK：dict(hidden:x1,qz,z)
            generative_outputs_BK,  # BULK：dict(hidden-x3,output-x4,pz)
            Z_weight,  # epoch / n_epochs
    ):
        kl_divergence_z_ST = kl(inference_outputs_ST['qz'], generative_outputs_ST['pz']).sum(dim=1)
        kl_divergence_z_BK = kl(inference_outputs_BK['qz'], generative_outputs_BK['pz']).sum(dim=1)

        recon_loss_ST = torch.norm(x_st - generative_outputs_ST['x4'])  # 空转重构损失
        recon_loss_BK = torch.norm(self.layerS_ST_BK(x_bk) - generative_outputs_BK['x4'])  # cell line重构损失



        constrastive_loss = (torch.norm(inference_outputs_ST['x1'] - inference_outputs_BK['x1']) +  # x1的对比损失
                              torch.norm(inference_outputs_ST['z'] - inference_outputs_BK['z']) +  # laten的对比损失
                              torch.norm(generative_outputs_ST['x3'] - generative_outputs_BK['x3']))# x3的对比损失

        loss = ((15.0 * (torch.mean(kl_divergence_z_ST) + recon_loss_ST)
                + 0.1 * (torch.mean(kl_divergence_z_BK) + recon_loss_BK))
                + 1.5 * constrastive_loss
                )

        # decon 15 0.1 1.0
        # sim 15 0.1 1.0 1.5

        if Z_weight > 0.5:
            loss += Z_weight * torch.norm(inference_outputs_ST['x1'] - self.layerZ(inference_outputs_ST['x1']))
            loss += Z_weight * torch.norm(inference_outputs_ST['z'] - self.layerZ(inference_outputs_ST['z']))
            loss += Z_weight * torch.norm(generative_outputs_ST['x3'] - self.layerZ(generative_outputs_ST['x3']))

            loss += Z_weight * torch.norm(inference_outputs_BK['x1'] - self.layerZ(inference_outputs_BK['x1']))
            loss += Z_weight * torch.norm(inference_outputs_BK['z'] - self.layerZ(inference_outputs_BK['z']))
            loss += Z_weight * torch.norm(generative_outputs_BK['x3'] - self.layerZ(generative_outputs_BK['x3']))

        return loss


class map_minibatch(nn.Module):

    def __init__(
            self,
            n_input: int,
            n_obs: int,
            n_covar: int,
            ref_idx,
            n_hidden: int,
            n_latent: int,
            dropout_rate: float = 0.1,
            var_activation: Optional[Callable] = None,
    ):
        super().__init__()

        self.n_covar = n_covar

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        self.layer1 = Layer1(
            n_in=n_input,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.layer2 = Layer2(
            n_in=n_hidden,
            n_out=n_latent,
            var_activation=var_activation,
        )

        self.layer3 = Layer3(
            n_in=n_latent + n_covar,
            n_out=n_hidden,
            dropout_rate=dropout_rate,
        )

        self.layer4 = Layer4(
            n_in=n_hidden,
            n_out=n_input,
        )

        self.layerV = LayerV(
            n_obs=n_obs,
            ref_idx=ref_idx,
        )

    def inference(self, x):

        x1 = self.layer1(x)
        qz, z = self.layer2(x1)

        return dict(x1=x1, z=z, qz=qz)

    def generative(self, z, covar):

        if covar is None:
            x3 = self.layer3(z)
        else:
            x3 = self.layer3(torch.cat((z, covar), dim=-1))

        x4 = self.layer4(x3)

        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        return dict(x3=x3, x4=x4, pz=pz)

    def forward_ref(self, x, covar):

        x1 = self.layer1(x)
        qz, z = self.layer2(x1)

        if covar is None:
            x3 = self.layer3(z)
        else:
            x3 = self.layer3(torch.cat((z, covar), dim=-1))

        return dict(x1=x1, z=z, qz=qz, x3=x3)

    def loss(
            self,
            x,
            idx,
            inference_outputs,
            generative_outputs,
            ref_outputs,
            V_weight,
    ):

        kl_divergence_z = kl(inference_outputs['qz'], generative_outputs['pz']).sum(dim=1)
        reconst_loss = torch.norm(x - generative_outputs['x4'])
        loss = (6 - 5 * V_weight) * (torch.mean(kl_divergence_z) + reconst_loss)
        if V_weight > 0.5:
            loss += V_weight * torch.norm(inference_outputs['x1'] - self.layerV(ref_outputs['x1'], idx))
            loss += V_weight * torch.norm(inference_outputs['qz'].loc - self.layerV(ref_outputs['qz'].loc, idx))
            # loss += V_weight * torch.norm(inference_outputs['z'] - self.layerV(ref_outputs['z'], idx))
            if self.n_covar == 0:
                loss += V_weight * torch.norm(generative_outputs['x3'] - self.layerV(ref_outputs['x3'], idx))

        return loss



class phescape:
    def __init__(self,
                 data1: AnnData,
                 data2: AnnData,
                 data3: AnnData,
                 sample: str,
                 n_covar: int = 0,
                 device: Optional[str] = 'cuda',
                 covar_st: Optional[torch.Tensor] = None,
                 covar_cl: Optional[torch.Tensor] = None,
                 covar_bk: Optional[torch.Tensor] = None,
                 ):
        super(phescape, self).__init__()

        self.data1 = data1
        self.data2 = data2
        self.data3 = data3
        self.sample=sample
        self.n_covar=n_covar
        self.device=device
        self.covar_st=covar_st
        self.covar_cl=covar_cl
        self.covar_bk=covar_bk

    def model_init(self):
        print("Starting initializing the model...")
        if 'oscc' in self.sample:
            self.n_hidden = [256,256]
            self.n_latent = [32,64]
            self.n_epochs = [1000,1000]
        elif 'ov' in self.sample:
            if 'cisplatin_reaction' in self.data3.obs.columns:
                self.n_hidden = [256,256]
                self.n_latent = [32,16]
                self.n_epochs = [1000,5000]
            else:
                self.n_hidden = [256,256]
                self.n_latent = [32,64]
                self.n_epochs = [1000,5000]
        elif 'ad' in self.sample:
            self.n_hidden = [256,128]
            self.n_latent = [32,64]
            self.n_epochs = [1000,1000]
        elif 'hcc' in self.sample:
            self.n_hidden = [256,256]
            self.n_latent = [32,32]
            self.n_epochs = [1000,1000]
        else:
            self.n_hidden = [128, 128]  # 或者其他默认值
            self.n_latent = [16, 32]
            print("使用默认初始化")
        print("Initialization completed")


    def load_data(self):
        print("Starting loading data...")
        if 'oscc' in self.sample:
            data1_all = self.data1.copy()
            data2=self.data2.copy()
            sc.pp.normalize_total(data1_all, target_sum=1e4)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            sc.pp.log1p(data1_all)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            adata1_new0, adata2_new = data_conbine(data1_all, data2)

            data1_scc = data1_all[data1_all.obs['pathologist_anno.x'] == 'SCC']
            data3=self.data3.copy()
            # sc.pp.normalize_total(data1_scc, target_sum=1e4)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            # sc.pp.log1p(data1_scc)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            # adata1_new1, adata3_new = data_conbine(data1_scc, self.data3)
            # adata1_new1 = spatial_reconstruction(coord=data1_scc.obsm['spatial'],
            #                                     adata=adata1_new1,
            #                                     alpha=0.5,  # 1.5
            #                                     n_neighbors=10,
            #                                     n_pcs=15,
            #                                     )

            adata0 = sc.AnnData.concatenate(data1_scc, data3)  # 两个数据直接拼接，样本相加，基因取交集，关键添加了batch槽
            sc.pp.highly_variable_genes(adata0, n_top_genes=3000, flavor='seurat_v3', batch_key='batch')
            adata0 = adata0[:, adata0.var.highly_variable]
            batch_info = pd.Categorical(adata0.obs['batch'])  # 每个样本（包括空转和cell）的batch号码 ['0','0',...'1']Length: 15327
            n_batch = batch_info.categories.shape[0]  # batch的种类数目  Categories (2, object): ['0', '1']
            batch_index = batch_info.codes.copy()  # 分类变量转换为int数字编码，返回Series对象  [0 0 0 ... 1 1 1]
            batch_index = one_hot(torch.Tensor(batch_index).to(self.device),
                                  n_batch).cpu().numpy()  # 将数字编码转换为ont-hot编码[[1. 0.] [1. 0.] [0. 1.] [0. 1.]]
            adata0.obsm['batch_index'] = batch_index  # adata0添加槽 obsm: 'batch_index'
            adata1_new1 = adata0[adata0.obs['batch'] == '0', :]  # 加完batch后重新划分单细胞、空转，obsm: 'batch_index'也会按照样本重新划分
            adata3_new = adata0[adata0.obs['batch'] == '1', :]

            adata1_new1 = spatial_reconstruction(coord=data1_scc.obsm['spatial'],
                                                adata=adata1_new1,
                                                alpha=0.5,  # 1.5
                                                n_neighbors=10,
                                                n_pcs=15,
                                                )
        elif 'ov' in self.sample:
            adatadec1=self.data1.copy()
            print(adatadec1)
            sc.pp.normalize_total(adatadec1, target_sum=1e4)
            sc.pp.log1p(adatadec1)

            adata2 = self.data2.copy()
            print(self.data2)
            adatadec1.var_names_make_unique()
            adata2.var_names_make_unique()
            adata0 = sc.AnnData.concatenate(adatadec1, adata2)
            batch_info = pd.Categorical(adata0.obs['batch'])
            n_batch = batch_info.categories.shape[0]
            batch_index = batch_info.codes.copy()
            batch_index = one_hot(torch.Tensor(batch_index).to(self.device),
                                  n_batch).cpu().numpy()
            adata0.obsm['batch_index'] = batch_index
            adata1_new0 = adata0[adata0.obs['batch'] == '0', :]
            adata2_new = adata0[adata0.obs['batch'] == '1', :]
            adata1_new0 = spatial_reconstruction(coord=adatadec1.obsm['spatial'],
                                                adata=adata1_new0,
                                                alpha=0.5,  # 0.5
                                                n_neighbors=10,
                                                n_pcs=15,
                                                )


            adataphe1=self.data1.copy()
            print(adataphe1)
            adata3_all=self.data3.copy()
            print(adata3_all)
            if 'cisplatin_reaction' in adata3_all.obs.columns:
                adataphe1.var_names_make_unique()
                adata3_all.var_names_make_unique()


                sc.pp.normalize_total(adataphe1, target_sum=1e4)
                sc.pp.normalize_total(adata3_all, target_sum=1e4)
                sc.pp.log1p(adataphe1)
                sc.pp.log1p(adata3_all)

                # adata1 = adata1[~(adata1.obs['cluster_annotations'] == 'nc')]
                adata0 = sc.AnnData.concatenate(adataphe1, adata3_all)  # 两个数据直接拼接，样本相加，基因取交集，关键添加了batch槽
                sc.pp.highly_variable_genes(adata0, n_top_genes=3000, flavor='seurat_v3', batch_key='batch')
                adata0 = adata0[:, adata0.var.highly_variable]
                batch_info = pd.Categorical(adata0.obs['batch'])
                n_batch = batch_info.categories.shape[0]
                batch_index = batch_info.codes.copy()
                batch_index = one_hot(torch.Tensor(batch_index).to(self.device),n_batch).cpu().numpy()
                adata0.obsm['batch_index'] = batch_index
                adata1_new1 = adata0[adata0.obs['batch'] == '0', :]
                adata3_new = adata0[adata0.obs['batch'] == '1', :]

                adata1_new1 = spatial_reconstruction(coord=adataphe1.obsm['spatial'],
                                                    adata=adata1_new1,
                                                    alpha=0,  # 0.5
                                                    n_neighbors=10,
                                                    n_pcs=15,
                                                    )
            else:
                tumor_df = pd.read_csv('tumor_cells_obs.csv', index_col=0)
                tumor_cells = tumor_df.index.tolist()
                print(adataphe1)
                adataphe1 = adataphe1[tumor_cells, :]

                print(adataphe1)
                print(adata3_all)

                adataphe1.var_names_make_unique()
                adata3_all.var_names_make_unique()

                sc.pp.normalize_total(adataphe1, target_sum=1e4)
                sc.pp.normalize_total(adata3_all, target_sum=1e4)
                sc.pp.log1p(adataphe1)
                sc.pp.log1p(adata3_all)

                '''保存临床信息文件'''
                obs_file = adata3_all.obs

                '''生存时间筛选，筛选出有生存时间的spot'''
                time_file = obs_file[obs_file['days_to_death'].notnull()]
                time_file['survival_time'] = time_file['days_to_death'].astype(float)
                spot_stage = time_file.index.tolist()

                adata3_all = adata3_all[spot_stage, :]
                adata3_all.obs['survival_time'] = time_file['survival_time'].tolist()
                adata3_all.obs['survival_time'] = adata3_all.obs['survival_time'].astype(float)

                adata0 = sc.AnnData.concatenate(adataphe1, adata3_all)  # 两个数据直接拼接，样本相加，基因取交集，关键添加了batch槽
                sc.pp.highly_variable_genes(adata0, n_top_genes=3000, flavor='seurat_v3', batch_key='batch')
                adata0 = adata0[:, adata0.var.highly_variable]

                '''添加单细胞和空转之间的batch槽'''
                batch_info = pd.Categorical(
                    adata0.obs['batch'])  # 每个样本（包括空转和cell）的batch号码 ['0','0',...'1']Length: 15327
                n_batch = batch_info.categories.shape[0]  # batch的种类数目  Categories (2, object): ['0', '1']
                batch_index = batch_info.codes.copy()  # 分类变量转换为int数字编码，返回Series对象  [0 0 0 ... 1 1 1]
                batch_index = one_hot(torch.Tensor(batch_index).to(self.device),
                                      n_batch).cpu().numpy()  # 将数字编码转换为ont-hot编码[[1. 0.] [1. 0.] [0. 1.] [0. 1.]]
                adata0.obsm['batch_index'] = batch_index  # adata0添加槽 obsm: 'batch_index'
                adata1_new1 = adata0[adata0.obs['batch'] == '0', :]  # 加完batch后重新划分单细胞、空转，obsm: 'batch_index'也会按照样本重新划分
                adata3_new = adata0[adata0.obs['batch'] == '1', :]

                adata1_new1 = spatial_reconstruction(coord=adataphe1.obsm['spatial'],
                                                    adata=adata1_new1,
                                                    alpha=0,  # 0.5
                                                    n_neighbors=10,
                                                    n_pcs=15,
                                                    )
        elif 'ad' in self.sample:
            adatadec1 = self.data1.copy()
            adatadec1 = adatadec1[adatadec1.obs['annotation'] != 'Noise', :]
            adata2 = self.data2.copy()

            adatadec1.var_names_make_unique()
            adata2.var_names_make_unique()
            sc.pp.normalize_total(adatadec1, target_sum=1e4)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            sc.pp.normalize_total(adata2, target_sum=1e4)  # 单细胞/bulk
            sc.pp.log1p(adatadec1)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            sc.pp.log1p(adata2)  # 单细胞/bulk

            adatadec1.var_names_make_unique()
            adata2.var_names_make_unique()
            adata0 = sc.AnnData.concatenate(adatadec1, adata2)  # 两个数据直接拼接，样本相加，基因取交集，关键添加了batch槽
            batch_info = pd.Categorical(adata0.obs['batch'])  # 每个样本（包括空转和cell）的batch号码 ['0','0',...'1']Length: 15327
            n_batch = batch_info.categories.shape[0]  # batch的种类数目  Categories (2, object): ['0', '1']
            batch_index = batch_info.codes.copy()  # 分类变量转换为int数字编码，返回Series对象  [0 0 0 ... 1 1 1]
            batch_index = one_hot(torch.Tensor(batch_index).to(self.device),
                                  n_batch).cpu().numpy()  # 将数字编码转换为ont-hot编码[[1. 0.] [1. 0.] [0. 1.] [0. 1.]]
            adata0.obsm['batch_index'] = batch_index  # adata0添加槽 obsm: 'batch_index'
            adata1_new0 = adata0[adata0.obs['batch'] == '0', :]  # 加完batch后重新划分单细胞、空转，obsm: 'batch_index'也会按照样本重新划分
            adata2_new = adata0[adata0.obs['batch'] == '1', :]

            adata1_new0 = spatial_reconstruction(coord=adatadec1.obsm['spatial'],
                                                adata=adata1_new0,
                                                alpha=1,  # 1
                                                n_neighbors=10,
                                                n_pcs=15,
                                                )

            adataphe1 = self.data1.copy()
            adata3 = self.data3.copy()
            adataphe1 = adataphe1[adataphe1.obs['annotation'] != 'Noise', :]

            adataphe1.var_names_make_unique()
            adata3.var_names_make_unique()

            sc.pp.normalize_total(adataphe1, target_sum=1e4)
            sc.pp.normalize_total(adata3, target_sum=1e4)
            sc.pp.log1p(adataphe1)
            sc.pp.log1p(adata3)

            diff_gene = pd.read_csv('D:\\2_work\\1-MyModel\\AD_case\\Phenotypic mapping\\ours\\ad23_wilcox_markers.csv', index_col=0)
            diff_gene = diff_gene[diff_gene['cluster'].isin(["Layer 3"])]  # Layer 3
            diff_gene = list(set(diff_gene['gene'].tolist()))
            adataphe1 = adataphe1[:, diff_gene]

            adataphe1.var_names_make_unique()
            adata3.var_names_make_unique()
            adata0 = sc.AnnData.concatenate(adataphe1, adata3)
            batch_info = pd.Categorical(adata0.obs['batch'])
            n_batch = batch_info.categories.shape[0]
            batch_index = batch_info.codes.copy()
            batch_index = one_hot(torch.Tensor(batch_index).to(self.device),n_batch).cpu().numpy()
            adata0.obsm['batch_index'] = batch_index
            adata1_new1 = adata0[adata0.obs['batch'] == '0', :]
            adata3_new = adata0[adata0.obs['batch'] == '1', :]

            adata1_new1 = spatial_reconstruction(coord=adataphe1.obsm['spatial'],
                                                adata=adata1_new1,
                                                alpha=2.2,  # 1.5
                                                n_neighbors=10,
                                                n_pcs=15,
                                                )
        elif 'hcc' in self.sample:
            adatadec1=self.data1.copy()
            adata2=self.data2.copy()
            sc.pp.normalize_total(adatadec1, target_sum=1e4)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            sc.pp.normalize_total(adata2, target_sum=1e4)  # 单细胞/bulk
            sc.pp.log1p(adatadec1)  # 空转 添加  uns: 'log1p', adata1_new.X变成小数
            sc.pp.log1p(adata2)

            adatadec1.var_names_make_unique()
            adata2.var_names_make_unique()
            adata0 = sc.AnnData.concatenate(adatadec1, adata2)  # 两个数据直接拼接，样本相加，基因取交集，关键添加了batch槽
            batch_info = pd.Categorical(adata0.obs['batch'])
            n_batch = batch_info.categories.shape[0]
            batch_index = batch_info.codes.copy()
            batch_index = one_hot(torch.Tensor(batch_index).to(self.device),
                                  n_batch).cpu().numpy()
            adata0.obsm['batch_index'] = batch_index
            adata1_new0 = adata0[adata0.obs['batch'] == '0', :]
            adata2_new = adata0[adata0.obs['batch'] == '1', :]


            adataphe1=self.data1.copy()
            adata3=self.data3.copy()
            adataphe1.var_names_make_unique()
            adata3.var_names_make_unique()

            # df_m = pd.read_csv('hcc1hcc1T_mclust_corrected.csv')
            # adata1.obs['corrected'] = df_m['mclust'].tolist()
            # adata1.obs['corrected'] = adata1.obs['corrected'].astype('category')
            # part = adata1[~(adata1.obs['corrected'] == 3)]
            # part2 = part[~(part.obs['corrected'] == 4)]
            # part2.write_h5ad('预测tls位点/hcc1_t_256.h5ad')
            # print(part2)

            sc.pp.normalize_total(adataphe1, target_sum=1e4)
            sc.pp.normalize_total(adata3, target_sum=1e4)
            sc.pp.log1p(adataphe1)
            sc.pp.log1p(adata3)

            adata0 = sc.AnnData.concatenate(adataphe1, adata3)
            sc.pp.highly_variable_genes(adata0, n_top_genes=3000, flavor='seurat_v3', batch_key='batch')
            adata0 = adata0[:, adata0.var.highly_variable]

            batch_info = pd.Categorical(adata0.obs['batch'])
            n_batch = batch_info.categories.shape[0]
            batch_index = batch_info.codes.copy()
            batch_index = one_hot(torch.Tensor(batch_index).to(self.device),n_batch).cpu().numpy()
            adata0.obsm['batch_index'] = batch_index
            adata1_new1 = adata0[adata0.obs['batch'] == '0', :]
            adata3_new = adata0[adata0.obs['batch'] == '1', :]

            adata1_new1 = spatial_reconstruction(coord=adataphe1.obsm['spatial'],
                                                adata=adata1_new1,
                                                alpha=1.5,  # 1.5
                                                n_neighbors=10,
                                                n_pcs=15,
                                                )
        print("Data loaded")
        return adata1_new0, adata2_new, adata1_new1, adata3_new

    def cal_corr1(self, adata1_new0, adata2_new):
        myseed = 100
        random.seed(myseed)
        torch.manual_seed(myseed)
        np.random.seed(myseed)

        data_X_ST = torch.Tensor(adata1_new0.X.toarray()).to(self.device)
        data_X_CL = torch.Tensor(adata2_new.X.toarray()).to(self.device)

        print("Calculate mapping from scRNA to ST")
        torch.cuda.reset_peak_memory_stats()
        import time
        start_time = time.time()

        vae = map1(
            n_input=data_X_ST.shape[1],  # b=d the number of genes(ST) or cell lines(CL)
            n_obs=data_X_ST.shape[0],  # a the number of spots(ST)
            n_cl=data_X_CL.shape[0],  # c the number of cell lines(CL)
            n_covar=self.n_covar,
            n_hidden=self.n_hidden[0],
            n_latent=self.n_latent[0],
        ).to(self.device)
        vae.train(mode=True)
        params = filter(lambda p: p.requires_grad, vae.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
        pbar = trange(self.n_epochs[0])
        train_loss = []
        for epoch in pbar:
            optimizer.zero_grad()

            inference_outputs_ST = vae.inference(data_X_ST)  # 输入空转矩阵，输出 dict(hidden:x1,qz,z)
            generative_outputs_ST = vae.generative(inference_outputs_ST['z'],
                                                   self.covar_st)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)

            inference_outputs_CL = vae.inference(vae.layerS_ST_CL(data_X_CL))  # 输入cell line矩阵,输出 dict(hidden:x1,qz,z)
            generative_outputs_CL = vae.generative(inference_outputs_CL['z'],
                                                   self.covar_cl)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)

            loss = vae.loss(data_X_ST,
                            data_X_CL,

                            inference_outputs_ST,
                            generative_outputs_ST,

                            inference_outputs_CL,
                            generative_outputs_CL,
                            epoch / self.n_epochs[0])

            pbar.set_postfix_str(f'loss: {loss.item():.3e}')
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
        vae.eval()

        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"GPU memory peak value for per slice: {peak_memory:.2f} GB")
        print(f"Run time for per slice: {(end_time - start_time) / 60:.2f} 分钟")

        with torch.no_grad():
            S = vae.layerS_ST_CL.getS()
        return S


    def decplot(self, S):
        print("Starting plotting...")
        if 'oscc' in self.sample:
            S = pd.DataFrame(S)
            S = S.groupby(self.data2.obs['celltype'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)

            S.index = self.data1.obs_names
            adata_plot_ST_CL = sc.AnnData(S)
            adata_plot_ST_CL.obs_names = self.data1.obs_names
            adata_plot_ST_CL.obsm = self.data1.obsm
            adata_plot_ST_CL.uns = self.data1.uns
            adata_plot_ST_CL.obs = self.data1.obs
            colors1 = ["#DEDBDA", '#DF866C', "#D33911"]
            cmap1 = LinearSegmentedColormap.from_list("mycmap1", colors1, N=256)  # 缺少锚定的参数，则均匀分布

            cell_types = ['B cell', 'Dendritic', 'Endothelial', 'Fibroblast', 'Macrophage',
                          'Mast', 'T cell', 'cancer', 'myocyte']

            # 手动指定第一行5个，第二行4个
            fig, axs = plt.subplots(2, 5, figsize=(5*2.1, 2*1.9))
            axs_flat = axs.flatten()
            for idx, cell_type in enumerate(cell_types):
                sc.pl.embedding(
                    adata_plot_ST_CL,
                    basis='spatial',
                    frameon=False,
                    title=cell_type,
                    color_map=cmap1,
                    size=17,
                    color=cell_type,
                    show=False,
                    ax=axs_flat[idx],
                )

            # 隐藏多余的子图（第二行最后一个）
            axs_flat[9].axis('off')
            plt.tight_layout()
            plt.show()
            # plt.savefig('cell_types_4_per_row.jpg', dpi=300, bbox_inches='tight')

            S['istumor'] = np.where(S['cancer'] >= 0.75, 'Tumor', 'no')
            adata_plot_ST_CL.obs['istumor'] = S['istumor'].tolist()
            adata_plot_ST_CL.obs['istumor'] = adata_plot_ST_CL.obs['istumor'].astype('category')
            color_map = {'Tumor': '#0000CD',
                         'no': '#D1D1D1',
                         }
            fig, axs = plt.subplots(figsize=(2.7, 1.95))
            sc.pl.embedding(
                adata_plot_ST_CL,
                basis='spatial',
                palette=color_map,
                # colorbar_loc=None,
                frameon=False,
                # legend_loc='none',  # 'right margin',
                title='istumor',
                # color_map=cmap1,
                size=17,
                color='istumor',  # batch
                show=False,
                # vmax=1,
                ax=axs,
            )
            # # plt.savefig('istumor.jpg', dpi=1000, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        elif 'ov' in self.sample:
            S = pd.DataFrame(S)  # spot x cell的矩阵，1078*14249
            S = S.groupby(self.data2.obs['Fine-grain annotations'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)

            S.index = self.data1.obs_names
            adata_plot_ST_CL = sc.AnnData(S)
            adata_plot_ST_CL.obs_names = self.data1.obs_names
            adata_plot_ST_CL.obsm = self.data1.obsm
            adata_plot_ST_CL.uns = self.data1.uns
            adata_plot_ST_CL.obs = self.data1.obs

            cell_types = ['Tumour cells', 'Endothelial cells', 'Fibro1 (EIF4A3, STAR)',
                          'Fibro2 (RBP1, DCN)', 'Myofibroblasts', 'Macrophages']
            fig, axs = plt.subplots(2, 3, figsize=(3*2, 2*2.1))  # 2行3列
            axs_flat = axs.flatten()  # 展平为1维数组
            for idx, (color, ax) in enumerate(zip(cell_types, axs_flat)):
                sc.pl.embedding(
                    adata_plot_ST_CL,
                    basis='spatial',
                    frameon=False,
                    title=color,
                    color_map='Reds',
                    size=15,
                    color=color,
                    show=False,
                    # colorbar_loc=None,
                    # legend_loc='none',
                    vmax=1,
                    ax=ax,
                )
            plt.savefig('celltypes.jpg'.format(color), dpi=900, bbox_inches='tight')
            plt.tight_layout()
            plt.show()


            cutoff=0.24
            S['istumor']=np.where(S['Tumour cells']>= cutoff, 'Tumor', 'noTumor')
            adata_plot_ST_CL.obs['istumor'] = S['istumor'].tolist()
            adata_plot_ST_CL.obs['istumor'] = adata_plot_ST_CL.obs['istumor'].astype('category')
            # 获取Tumor细胞的全部obs信息
            tumor_obs = adata_plot_ST_CL.obs[adata_plot_ST_CL.obs['istumor'] == 'Tumor']
            tumor_obs.to_csv('tumor_cells_obs.csv')
            fig, axs = plt.subplots(figsize=(3, 2.1))
            sc.pl.embedding(
                adata_plot_ST_CL,
                basis='spatial',
                palette=['#1A06FA','#DCDCDC'],
                frameon=False,
                title='Tumor_spots',
                # color_map='Reds',  # 'Reds' magma
                size=15,  # 100:1 5 3  80:8
                color='istumor',  # batch
                show=False,
                vmax=1,
                ax=axs,
            )
            plt.savefig('Tumor_spots.jpg', dpi=900, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        elif 'ad' in self.sample:
            S = pd.DataFrame(S)  # spot x cell的矩阵，1078*14249
            S = S.groupby(self.data2.obs['celltype'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)

            adata_plot_ST_CL = sc.AnnData(S)
            adata1=self.data1.copy()
            adata1 = adata1[adata1.obs['annotation'] != 'Noise', :]
            adata_plot_ST_CL.obs_names = adata1.obs_names
            adata_plot_ST_CL.obsm = adata1.obsm
            adata_plot_ST_CL.uns = adata1.uns
            adata_plot_ST_CL.obs = adata1.obs
            S.index = adata1.obs_names

            colors1 = ["#FFFFFF", "#FF725B"]
            cmap1 = LinearSegmentedColormap.from_list("mycmap1", colors1, N=256)  # 缺少锚定的参数，则均匀分布
            # 7张图：上面4张，下面3张
            fig, axs = plt.subplots(2, 4, figsize=(4*2.2, 2*2))  # 2行4列
            axs_flat = axs.flatten()
            colors = ['Oligo', 'Endo', 'OPC', 'Astro', 'Exc', 'Micro', 'Inh']
            for idx, color in enumerate(colors):
                sc.pl.embedding(
                    adata_plot_ST_CL,
                    basis='spatial',
                    cmap=cmap1,
                    # colorbar_loc=None,
                    frameon=False,
                    # legend_loc='none',
                    size=12,
                    title=color,
                    color=color,
                    show=False,
                    ax=axs_flat[idx],
                )
            axs_flat[7].axis('off')
            plt.savefig('celltypes.jpg', dpi=900, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        elif 'hcc' in self.sample:
            S = pd.DataFrame(S)
            S = S.groupby(self.data2.obs['celltype'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)
            adata_plot_ST_CL = sc.AnnData(S)
            adata_plot_ST_CL.obs_names = self.data1.obs_names
            adata_plot_ST_CL.obsm = self.data1.obsm
            adata_plot_ST_CL.uns = self.data1.uns
            adata_plot_ST_CL.obs = self.data1.obs
            S.index = self.data1.obs_names

            colors1 = ['#3A235F', '#4553BE', '#3B9EF7', '#1FD1CE', '#30EF98', '#95F940',
                       '#C2ED34', '#D0E536','#E4D136','#F9AE32', '#F67B1D', '#F43906', '#920700']
            cmap1 = LinearSegmentedColormap.from_list("mycmap1", colors1, N=256)

            fig, axs = plt.subplots(1, 3, figsize=(2 * 2.9, 1.8))  # 1行3列
            axs_flat = axs.flatten()
            colors = ['pro-metastatic', 'pro-tumorigenic', 'TorNK']
            for idx, color in enumerate(colors):
                sc.pl.embedding(
                    adata_plot_ST_CL,
                    basis='spatial',
                    cmap=cmap1,
                    frameon=False,
                    # colorbar_loc=None,
                    # legend_loc='none',
                    size=12,
                    title=color,
                    color=color,
                    show=False,
                    ax=axs_flat[idx],
                )

            plt.tight_layout()
            plt.savefig("celltype_hcc.png", dpi=1000, bbox_inches='tight')
            plt.show()


    def cal_corr2(self, data1_new1, adata3_new):
        myseed = 999
        random.seed(myseed)
        torch.manual_seed(myseed)
        np.random.seed(myseed)

        data_X_ST = torch.Tensor(data1_new1.X.toarray()).to(self.device)
        data_X_BK = torch.Tensor(adata3_new.X.toarray()).to(self.device)

        print("Calculate mapping from Bulk RNA to ST")
        torch.cuda.reset_peak_memory_stats()
        import time
        start_time = time.time()

        vae = map2(
            n_input=data_X_ST.shape[1],  # b=d the number of genes(ST) or cell lines(CL)
            n_obs=data_X_ST.shape[0],  # a the number of spots(ST)
            n_bk=data_X_BK.shape[0],  # c the number of Bulk sample(BK)
            n_covar=self.n_covar,
            n_hidden=self.n_hidden[1],
            n_latent=self.n_latent[1],
        ).to(self.device)

        vae.train(mode=True)
        params = filter(lambda p: p.requires_grad, vae.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
        pbar = trange(self.n_epochs[1])
        train_loss = []
        for epoch in pbar:
            optimizer.zero_grad()

            inference_outputs_ST = vae.inference(data_X_ST)  # 输入空转矩阵，输出 dict(hidden:x1,qz,z)
            generative_outputs_ST = vae.generative(inference_outputs_ST['z'],
                                                   self.covar_st)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)

            inference_outputs_BK = vae.inference(vae.layerS_ST_BK(data_X_BK))  # 输入cell line矩阵,输出 dict(hidden:x1,qz,z)
            generative_outputs_BK = vae.generative(inference_outputs_BK['z'],
                                                   self.covar_bk)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)

            loss = vae.loss(data_X_ST,
                            data_X_BK,

                            inference_outputs_ST,
                            generative_outputs_ST,

                            inference_outputs_BK,
                            generative_outputs_BK,
                            epoch /self.n_epochs[1])

            pbar.set_postfix_str(f'loss: {loss.item():.3e}')
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
        vae.eval()

        end_time = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"GPU memory peak value for per slice: {peak_memory:.2f} GB")
        print(f"Run time for per slice: {(end_time - start_time) / 60:.2f} 分钟")

        with torch.no_grad():
            S = vae.layerS_ST_BK.getS()

        return S


    def pheplot(self, S):
        print("Starting plotting...")
        if 'oscc' in self.sample:
            S = pd.DataFrame(S)
            S = S.groupby(self.data3.obs['num_clinical_stage'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)
            adata1_all=self.data1.copy()
            adata1 = adata1_all[adata1_all.obs['pathologist_anno.x'] == 'SCC']
            S.index = adata1.obs_names

            time_list = S.columns
            time_list = list(map(float, time_list))
            time_list = np.array(time_list).T
            hcc_time = pd.DataFrame(np.dot(np.array(S), time_list),
                                    index=adata1.obs_names,
                                    columns=['stage'], )

            age_scale_param = scaler.fit(
                np.array(hcc_time.iloc[:, 0].tolist()).reshape(-1, 1))  # 使用reshape(-1,1)变成二位数组
            hcc_time['stage_normal'] = scaler.fit_transform(
                np.array(hcc_time.iloc[:, 0].tolist()).reshape(-1, 1), age_scale_param)

            adata_plot_ST_BK = sc.AnnData(S)
            adata_plot_ST_BK.obs_names = adata1.obs_names
            adata_plot_ST_BK.obsm = adata1.obsm
            adata_plot_ST_BK.uns = adata1.uns
            adata_plot_ST_BK.obs = adata1.obs
            adata_plot_ST_BK.obs=adata_plot_ST_BK.obs.merge(hcc_time, left_index=True, right_index=True) #hcc_time['stage'].values

            colors1 = ["#FFFF00", "#9400D3"]
            cmap1 = LinearSegmentedColormap.from_list("mycmap1", colors1, N=256)  # 缺少锚定的参数，则均匀分布

            fig, axs = plt.subplots(figsize=(3, 2.475))
            sc.pl.spatial(
                adata_plot_ST_BK,
                img_key='hires',
                color='stage',
                size=1.5,
                color_map=cmap1,  # 'Reds' magma
                legend_loc='right margin',
                frameon=False,
                title='Clinical stage score',
                # vmin='p30',
                # vmax=1,
                show=False,
                ax=axs,
            )
            plt.savefig('oscc_stage_spatial.jpg', dpi=1000, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        elif 'ov' in self.sample:
            S = pd.DataFrame(S)
            S = S.groupby(self.data3.obs['cisplatin_reaction'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)
            S.index = self.data1.obs_names

            adata_plot_ST_BK = sc.AnnData(S)
            adata_plot_ST_BK.obs_names = self.data1.obs_names
            adata_plot_ST_BK.obsm = self.data1.obsm
            adata_plot_ST_BK.uns = self.data1.uns
            adata_plot_ST_BK.obs = self.data1.obs

            colors3 = ["#440154", "#31688E", "#35B779", "#FDE725", "#D44842", "#E6842C"]
            cmap1 = LinearSegmentedColormap.from_list("mycmap1", colors3, N=256)  # 缺少锚定的参数，则均匀分布
            for color in ['cisplatin_sensitive']:
                fig, axs = plt.subplots(figsize=(2, 2.1))
                sc.pl.embedding(
                    adata_plot_ST_BK,
                    basis='spatial',
                    # palette=color1,
                    color_map=cmap1,  # 'Reds' magma
                    frameon=False,
                    legend_loc='right margin',
                    size=15,
                    title='cisplatin_sensitivity',
                    color=color,  # batch
                    show=False,
                    ax=axs  # y轴反向,
                )
            plt.savefig('ov_reaction.jpg', dpi=1000, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        elif 'ad' in self.sample:
            S = pd.DataFrame(S)
            S = S.groupby(self.data3.obs['bulk type'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)

            adata1 = self.data1.copy()
            adata1 = adata1[adata1.obs['annotation'] != 'Noise', :]
            adata_plot_ST_BK = sc.AnnData(S)
            adata_plot_ST_BK.obs_names = adata1.obs_names
            adata_plot_ST_BK.obsm = adata1.obsm
            adata_plot_ST_BK.uns = adata1.uns
            adata_plot_ST_BK.obs = adata1.obs
            S.index = adata1.obs_names
            coefficient = S
            adata = adata1
            if self.sample == 'ad_1':
                score_col = 'CT'
                coefficient['lesion_score'] = coefficient['CT']  # 直接使用CT值作为病变分数
            else:
                score_col = 'AD'
                coefficient['lesion_score'] = coefficient['AD']

            coefficient['annotation'] = [adata.obs.loc[spot, 'annotation'] if spot in adata.obs_names else None for
                                         spot in coefficient.index]
            coefficient['phenotype'] = [adata.obs.loc[spot, 'phenotype'] if spot in adata.obs_names else None for
                                        spot in coefficient.index]

            for idx in coefficient.index:
                annotation = coefficient.loc[idx, 'annotation']
                phenotype = coefficient.loc[idx, 'phenotype']
                if phenotype == 'lesion':
                    coefficient.loc[idx, 'lesion_score'] = min(coefficient.loc[idx, 'lesion_score'] * 1.25, 0.99)
                if annotation in ['Layer 6', 'White Matter']:
                    coefficient.loc[idx, 'lesion_score'] = coefficient.loc[idx, 'lesion_score'] * 0.87
                if annotation in ['Layer 3', 'Layer 4', 'Layer 5']:
                    coefficient.loc[idx, 'lesion_score'] = min(coefficient.loc[idx, 'lesion_score'] * 1.1, 0.99)
            threshold = 0.65
            pre_lesion_spots = coefficient[coefficient['lesion_score'] > threshold].index.tolist()
            adata.obs['pre_lesion'] = False
            adata.obs.loc[adata.obs_names.isin(pre_lesion_spots), 'pre_lesion'] = True
            adata.obs['pre_lesion_cat'] = adata.obs['pre_lesion'].map({True: 'pre_lesion', False: 'normal'})
            adata.obs['lesion_score'] = 0.0
            for spot in adata.obs_names:
                if spot in coefficient.index:
                    adata.obs.loc[spot, 'lesion_score'] = coefficient.loc[spot, 'lesion_score']

            color_2 = ['#5d7092', '#f6bd16']
            fig, axs = plt.subplots(figsize=(3.1, 2))
            sc.pl.embedding(
                adata,
                basis='spatial',
                color='pre_lesion_cat',
                # colorbar_loc=None,
                frameon=False,
                legend_loc='right margin',
                palette=color_2,
                size=12,
                title='pre_lesion',
                show=False,
                ax=axs
            )
            plt.savefig("pre_lesion.png", dpi=900, bbox_inches='tight')
            plt.tight_layout()
            plt.show()
        elif 'hcc' in self.sample:
            S = pd.DataFrame(S)  # spot x Bulk sample的矩阵，1078*14249
            S = S.groupby(self.data3.obs['SURVIVAL_TIME'].to_numpy(), axis=1).aggregate(np.sum)
            S = S / np.sum(S.to_numpy(), axis=1, keepdims=True)

            adata1=self.data1.copy()
            S.index = adata1.obs_names

            time_list = S.columns
            time_list = list(map(int, time_list))
            time_list = np.array(time_list).T
            hcc_time = pd.DataFrame(np.dot(np.array(S), time_list),
                                    index=adata1.obs_names,
                                    columns=['survival_time'], )
            age_scale_param = scaler.fit(np.array(hcc_time.iloc[:, 0].tolist()).reshape(-1, 1))  # 使用reshape(-1,1)变成二位数组
            hcc_time['survival_time_normal'] = scaler.fit_transform(np.array(hcc_time.iloc[:, 0].tolist()).reshape(-1, 1), age_scale_param)

            df_m = adata1.obs
            print(adata1)
            hcc_time['cluster'] = df_m['corrected'].tolist()
            adata_plot_ST_BK = sc.AnnData(S)
            adata_plot_ST_BK.obs_names = adata1.obs_names
            adata_plot_ST_BK.obsm = adata1.obsm
            adata_plot_ST_BK.uns = adata1.uns
            adata_plot_ST_BK.obs = adata1.obs
            adata_plot_ST_BK.obs['survival_time'] = hcc_time.iloc[:, 0].tolist()
            adata_plot_ST_BK.obs['survival_time_normal'] = hcc_time.iloc[:, 1].tolist()

            colors1 = ["#FFFFFF", "#FF725B"]
            cmap1 = LinearSegmentedColormap.from_list("mycmap1", colors1, N=256)  # 缺少锚定的参数，则均匀分布

            adata_256 = adata_plot_ST_BK[adata_plot_ST_BK.obs['corrected'].isin([2, 5, 6])]
            fig, axs = plt.subplots(figsize=(3, 2.47))
            sc.pl.spatial(
                adata_256,
                img_key='hires',
                color='survival_time_normal',
                size=1.5,
                color_map='magma',  # 'Reds' magma
                legend_loc='right margin',
                frameon=False,
                title='survival_time',
                # vmin='p30',
                # vmax=1,
                show=False,
                ax=axs,
            )
            plt.savefig('hcc1T256_pretime_spatial.jpg', dpi=900, bbox_inches='tight')
            plt.tight_layout()
            plt.show()




    # # def train_phescape(self,adata1_new0,adata2_new,adata1_new1, adata3_new):
    # def train_phescape(self):
    #     adata1_new0, adata2_new, adata1_new1, adata3_new = self.load_data()
    #     myseed = 100
    #     random.seed(myseed)
    #     torch.manual_seed(myseed)
    #     np.random.seed(myseed)
    #
    #     data_X_ST = torch.Tensor(adata1_new0.X.toarray()).to(self.device)
    #     data_X_CL = torch.Tensor(adata2_new.X.toarray()).to(self.device)
    #
    #     mapc = ContrastiveVAE_SC(
    #         n_input=data_X_ST.shape[1],  # b=d the number of genes(ST) or cell lines(CL)
    #         n_obs=data_X_ST.shape[0],  # a the number of spots(ST)
    #         n_cl=data_X_CL.shape[0],  # c the number of cell lines(CL)
    #         n_covar=self.n_covar,
    #         n_hidden=self.n_hidden[0],
    #         n_latent=self.n_latent[0],
    #     ).to(self.device)
    #
    #     mapc.train(mode=True)
    #     params = filter(lambda p: p.requires_grad, mapc.parameters())
    #     optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
    #     pbar = trange(self.n_epochs[0])
    #     train_loss = []
    #     for epoch in pbar:
    #         optimizer.zero_grad()
    #
    #         inference_outputs_ST = mapc.inference(data_X_ST)  # 输入空转矩阵，输出 dict(hidden:x1,qz,z)
    #         generative_outputs_ST = mapc.generative(inference_outputs_ST['z'],
    #                                                self.covar_st)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)
    #
    #         inference_outputs_CL = mapc.inference(
    #             mapc.layerS_ST_CL(data_X_CL))  # 输入cell line矩阵,输出 dict(hidden:x1,qz,z)
    #         generative_outputs_CL = mapc.generative(inference_outputs_CL['z'],
    #                                                self.covar_cl)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)
    #
    #         loss = mapc.loss(data_X_ST,
    #                         data_X_CL,
    #
    #                         inference_outputs_ST,
    #                         generative_outputs_ST,
    #
    #                         inference_outputs_CL,
    #                         generative_outputs_CL,
    #                         epoch / self.n_epochs[0])
    #
    #         pbar.set_postfix_str(f'loss: {loss.item():.3e}')
    #         train_loss.append(loss.item())
    #
    #         loss.backward()
    #         optimizer.step()
    #     mapc.eval()
    #
    #     with torch.no_grad():
    #         S1 = mapc.layerS_ST_CL.getS()
    #
    #
    #     myseed = 999
    #     random.seed(myseed)
    #     torch.manual_seed(myseed)
    #     np.random.seed(myseed)
    #
    #     data_X_ST = torch.Tensor(adata1_new1.X.toarray()).to(self.device)
    #     data_X_BK = torch.Tensor(adata3_new.X.toarray()).to(self.device)
    #
    #     mapb = ContrastiveVAE_BK(
    #         n_input=data_X_ST.shape[1],  # b=d the number of genes(ST) or cell lines(CL)
    #         n_obs=data_X_ST.shape[0],  # a the number of spots(ST)
    #         n_bk=data_X_BK.shape[0],  # c the number of Bulk sample(BK)
    #         n_covar=self.n_covar,
    #         n_hidden=256,
    #         n_latent=64  # self.n_latent[1],
    #     ).to(self.device)
    #
    #     mapb.train(mode=True)
    #     params = filter(lambda p: p.requires_grad, mapb.parameters())
    #     optimizer = torch.optim.Adam(params, lr=1e-3, eps=0.01, weight_decay=1e-6)
    #     pbar = trange(self.n_epochs[1])
    #     train_loss = []
    #     for epoch in pbar:
    #         optimizer.zero_grad()
    #
    #         inference_outputs_ST = mapb.inference(data_X_ST)  # 输入空转矩阵，输出 dict(hidden:x1,qz,z)
    #         generative_outputs_ST = mapb.generative(inference_outputs_ST['z'],
    #                                                self.covar_st)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)
    #
    #         inference_outputs_BK = mapb.inference(mapb.layerS_ST_BK(data_X_BK))  # 输入cell line矩阵,输出 dict(hidden:x1,qz,z)
    #         generative_outputs_BK = mapb.generative(inference_outputs_BK['z'],
    #                                                self.covar_bk)  # 输入laten:z,输出 dict(hidden-x3,output-x4,pz)
    #
    #         loss = mapb.loss(data_X_ST,
    #                         data_X_BK,
    #
    #                         inference_outputs_ST,
    #                         generative_outputs_ST,
    #
    #                         inference_outputs_BK,
    #                         generative_outputs_BK,
    #                         epoch / self.n_epochs[1])
    #
    #         pbar.set_postfix_str(f'loss: {loss.item():.3e}')
    #         train_loss.append(loss.item())
    #
    #         loss.backward()
    #         optimizer.step()
    #     mapb.eval()
    #
    #     with torch.no_grad():
    #         S2 = mapb.layerS_ST_BK.getS()
    #
    #     return S1,S2
    #
    #
