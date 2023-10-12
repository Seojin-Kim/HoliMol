import argparse
import time
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import GNN
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, GlobalAttention
from torch_scatter import scatter

from datasets import MoleculeDataset_graphcl


class graphcl3d_brics(nn.Module):
    def __init__(self, gnn):
        super(graphcl3d_brics, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300))
        self.aux = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1),
            nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(600, 300), nn.ReLU(inplace=True), nn.Linear(300,1))
        self.ce = nn.CrossEntropyLoss(reduction = 'none')
        self.bce = nn.BCEWithLogitsLoss()

    def forward_cl(self, x, position, batch):
        x = self.gnn(x, position, batch)
        x = self.projection_head(x)
        return x
    def forward_mae(self, x, position, batch):
        x = x.type(torch.LongTensor).cuda()

        x = self.gnn(x, position, batch)
        return x
    
    def forward_frag_cl(self, frag0, frag1, frag2, frag3):

        x0 = self.gnn(frag0.x[:,0], frag0.positions+ torch.randn_like(frag0.positions).cuda(), frag0.batch)
        x1 = self.gnn(frag1.x[:,0], frag1.positions+ torch.randn_like(frag1.positions).cuda(), frag1.batch)
        x2 = self.gnn(frag2.x[:,0], frag2.positions+ torch.randn_like(frag2.positions).cuda(), frag2.batch)
        x3 = self.gnn(frag3.x[:,0], frag3.positions+ torch.randn_like(frag3.positions).cuda(), frag3.batch)
        

        x = (x0 + x1 + x2 + x3)/4.0

        x = self.projection_head(x)
        
        return x
    

    def loss_cl(self, x1, x2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)


        sim_matrix = torch.exp(sim_matrix / T)

        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = (pos_sim) / (sim_matrix.sum(dim=1) - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss


class graphcl3d(nn.Module):
    def __init__(self, gnn):
        super(graphcl3d, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300))
        self.aux = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1),
            nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(600, 300), nn.ReLU(inplace=True), nn.Linear(300,1))
        self.ce = nn.CrossEntropyLoss(reduction = 'none')
        self.bce = nn.BCEWithLogitsLoss()
        
        self.attention_pool = GlobalAttention(gate_nn = torch.nn.Linear(300, 1))

    def forward_cl_node(self, x, position, batch):
        node_z, x = self.gnn(x, position, batch, node_feat = True)
        x = self.projection_head(x)
        return node_z, x
    
    def forward_cl(self, x, position, batch):
        
        x = self.gnn(x, position, batch)
        x = self.projection_head(x)
        return x
    
    def average_pool(self, x, batch):
        x = self.pool(x, batch)
        return x

    def loss_cl_graphcl(self, x1, x2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)


        sim_matrix = torch.exp(sim_matrix / T)

        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = (pos_sim) / (sim_matrix.sum(dim=1) - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss
    
    def forward_project(self, x):
        x = self.projection_head(x)
        return x    
    
    def attention(self, x, batch):
        x = self.attention_pool(x, batch)
        return x
    
    def forward_pool(self, x, position, batch):
        x = self.gnn(x, position, batch)
        return x
    
    def weighted_forward_frag_cl(self, x1, position1, batch1, x2, position2, batch2):
        x1 = self.gnn(x1, position1, batch1)

        x2 = self.gnn(x2, position2, batch2)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        x = x1 * batch1_ratio.unsqueeze(1).repeat(1,x1.size(1)) + x2 * batch2_ratio.unsqueeze(1).repeat(1,x2.size(1))

        x = self.projection_head(x)
        frag1 = self.projection_head(x1)
        frag2 = self.projection_head(x2)
        return x, frag1, frag2
    

    def weighted_loss_cl_onlyneg(self, x1, x2, frag1, frag2, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()

        #batch1_pos = (batch1_ratio>0.7).type(torch.FloatTensor).cuda()
        #batch2_pos = (batch2_ratio>0.7).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = (pos_sim) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss



class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300))
        self.aux = nn.Sequential(
            nn.Linear(900, 450),
            nn.ReLU(inplace=True),
            nn.Linear(450, 9))
        self.aux2 = nn.Sequential(
            nn.Linear(1200, 600),
            nn.ReLU(inplace=True),
            nn.Linear(600, 18))
#         self.aux2 = nn.Linear(1200, 18, bias = True)
        self.aux_united = nn.Sequential(
            nn.Linear(300, 150),
            nn.ReLU(inplace=True),
            nn.Linear(150, 18))

        self.aux_inner = nn.Sequential(
            nn.Linear(600, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 150))
        self.aux_tail = nn.Sequential(
            nn.Linear(600, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 150))
        self.aux_aug = nn.Sequential(nn.Linear(300,18))
        self.classifier = nn.Sequential(nn.Linear(600, 300), nn.ReLU(inplace=True), nn.Linear(300,1))
        self.ce = nn.CrossEntropyLoss()
        self.ce2 = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.add_pool = global_add_pool
        self.attention_pool = GlobalAttention(gate_nn = torch.nn.Linear(300,1))

    def forward_cl_aux(self, x, edge_index, edge_attr, batch, positions):
        x = self.gnn(x, edge_index, edge_attr)
        gen_p = self.aux(x)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x, gen_p
    
    def forward_aux(self, x):
        x = self.aux(x)
        return x
    def forward_aux2(self, x):
        x = self.aux2(x)
        return x
    
    def average_pool(self, x, batch):
        x = self.pool(x, batch)
        return x
    
    def attention(self, x, batch):
        x = self.attention_pool(x, batch)
        return x
    
    def forward_pool(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        return x
    
    def forward_project(self, x):
        x = self.projection_head(x)
        return x    
    
    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def forward_cl_node(self, x, edge_index, edge_attr, batch):
        node_rep = self.gnn(x, edge_index, edge_attr)
        mol_rep = self.pool(node_rep, batch)
        x = self.projection_head(mol_rep)
        return node_rep, mol_rep, x
    
    def forward_frag_cl(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2):
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        x = (x1+x2)/2.0

        x = self.projection_head(x)
        frag1 = self.projection_head(x1)
        frag2 = self.projection_head(x2)
        return x, frag1, frag2

    def forward_frag_cl_old(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2):
        
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        x = (x1+x2)/2.0

        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2, frag1, frag2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)
        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                     torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                     torch.einsum('i,j->ij', x1_abs, frag2_abs)

        sim_matrix = torch.exp(sim_matrix / T)
        
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 + neg2 - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss


    def loss_cl_graphcl(self, x1, x2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)
        
        sim_matrix = torch.exp(sim_matrix / T)
        
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss

    def loss_cl_fp(self, x1, x2, frag1, frag2, fps, thres):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)
        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                     torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                     torch.einsum('i,j->ij', x1_abs, frag2_abs)

        sim_matrix = torch.exp(sim_matrix / T)
        fp_inner = torch.matmul(fps, fps.transpose(0,1))
        fp_sum = torch.sum(fps, dim=1).unsqueeze(0)
        tanimoto_sim_matrix = fp_inner / ((fp_sum + fp_sum.transpose(0,1))-fp_inner)
        
        sim_matrix = sim_matrix * ((tanimoto_sim_matrix<thres).type(torch.float) + torch.eye(batch).cuda())
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 + neg2 - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss


    def loss_cl_fp_onlyneg(self, x1, x2, frag1, frag2, fps, thres, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)
        
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)
        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                     torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                     torch.einsum('i,j->ij', x1_abs, frag2_abs)

        
        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()
        
        sim_matrix = torch.exp(sim_matrix / T)
        fp_inner = torch.matmul(fps, fps.transpose(0,1))
        fp_sum = torch.sum(fps, dim=1).unsqueeze(0)
        tanimoto_sim_matrix = fp_inner / ((fp_sum + fp_sum.transpose(0,1))-fp_inner)
        
        sim_matrix = sim_matrix * ((tanimoto_sim_matrix<thres).type(torch.float) + torch.eye(batch).cuda())
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss

    def loss_cl_old(self, x1, x2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss

    def weighted_forward(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2):
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        x = x1 * batch1_ratio.unsqueeze(1).repeat(1,x1.size(1)) + x2 * batch2_ratio.unsqueeze(1).repeat(1,x2.size(1))

        return x

    def weighted_forward_frag_cl(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2):
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        x = x1 * batch1_ratio.unsqueeze(1).repeat(1,x1.size(1)) + x2 * batch2_ratio.unsqueeze(1).repeat(1,x2.size(1))

        x = self.projection_head(x)
        frag1 = self.projection_head(x1)
        frag2 = self.projection_head(x2)
        return x, frag1, frag2

    def weighted_forward_frag_cl2(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2):
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        num_to_permute = int(np.ceil(x1.shape[0] * 0.3))
        
        indices_to_permute = torch.randperm(x1.shape[0], device = x1.device)[:num_to_permute]
        tmp = torch.cat([x2, batch2_numatoms.unsqueeze(-1)], dim = -1)
        tmp2 = tmp.clone()
        is_permuted = torch.zeros(x1.shape[0], dtype = int, device = x1.device)
        for i in indices_to_permute:
            is_permuted[i] = 1
            if i != x1.shape[0] - 1:
                tmp2[i] = tmp[i+1]
            else:
                tmp2[i] = tmp[0]

        x_mix = tmp2
        mix_numatoms = x_mix[:,-1]
        total_numatoms2 = batch1_numatoms + mix_numatoms
        batch1_ratio2 = batch1_numatoms/total_numatoms2
        mix_ratio = mix_numatoms/total_numatoms2
        x_mix = x_mix[:,:-1]
        neg = x1 * batch1_ratio2.unsqueeze(1).repeat(1,x1.size(1)) + x_mix * mix_ratio.unsqueeze(1).repeat(1,x_mix.size(1))

        x = x1 * batch1_ratio.unsqueeze(1).repeat(1,x1.size(1)) + x2 * batch2_ratio.unsqueeze(1).repeat(1,x2.size(1))

        x = self.projection_head(x)

        frag1 = self.projection_head(x1)
        frag2 = self.projection_head(x2)
        return x, frag1, frag2, neg, is_permuted

    def weighted_forward_frag_cl3(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2):
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        num_to_permute = int(np.ceil(x1.shape[0] * 0.5))
        
        indices_to_permute = torch.randperm(x1.shape[0], device = x1.device)[:num_to_permute]
        #tmp = torch.cat([x2, batch2_numatoms.unsqueeze(-1)], dim = -1)
        x2_permuted = x2.clone().detach()
        is_permuted = torch.zeros(x1.shape[0], dtype = int, device = x1.device)
        for i, idx in enumerate(indices_to_permute):
            is_permuted[idx] = 1
            x2_permuted[idx] = x2[indices_to_permute[0]].detach()#x2[indices_to_permute[i-1]]

        unmixed_x = torch.cat([x1, x2], dim=1)
        mixed_x = torch.cat([x1, x2_permuted], dim=1)

        x = x1 * batch1_ratio.unsqueeze(1).repeat(1,x1.size(1)) + x2 * batch2_ratio.unsqueeze(1).repeat(1,x2.size(1))

        x = self.projection_head(x)

        frag1 = self.projection_head(x1)
        frag2 = self.projection_head(x2)
        return x, frag1, frag2, unmixed_x, mixed_x, is_permuted

    def weighted_forward_frag_cl4(self, x1, edge_index1, edge_attr1, batch1, x2, edge_index2, edge_attr2, batch2):
        x1 = self.gnn(x1, edge_index1, edge_attr1)
        x1 = self.pool(x1, batch1)

        x2 = self.gnn(x2, edge_index2, edge_attr2)
        x2 = self.pool(x2, batch2)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        num_to_permute = int(np.ceil(x1.shape[0] * 0.5))
        
        indices_to_permute = torch.randperm(x1.shape[0], device = x1.device)[:num_to_permute]
        #tmp = torch.cat([x2, batch2_numatoms.unsqueeze(-1)], dim = -1)
        x2_permuted = x2.clone()
        is_permuted = torch.zeros(x1.shape[0], dtype = int, device = x1.device)
        for i, idx in enumerate(indices_to_permute):
            is_permuted[idx] = 1
            x2_permuted[idx] = x2[indices_to_permute[i-1]]#x2[indices_to_permute[i-1]]

        #unmixed_x = torch.cat([x1, x2], dim=1)
        #mixed_x = torch.cat([x1, x2_permuted], dim=1)

        x = x1 * batch1_ratio.unsqueeze(1).repeat(1,x1.size(1)) + x2 * batch2_ratio.unsqueeze(1).repeat(1,x2.size(1))

        x = self.projection_head(x)

        frag1 = self.projection_head(x1)
        frag2 = self.projection_head(x2)
        return x, frag1, frag2, x2_permuted, is_permuted


    def weighted_loss_cl(self, x1, x2, frag1, frag2, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.6).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.6).type(torch.FloatTensor).cuda()

        batch1_pos = (batch1_ratio>0.9).type(torch.FloatTensor).cuda()
        batch2_pos = (batch2_ratio>0.9).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        # loss = (pos_sim + neg1 * batch1_pos + neg2 * batch2_pos) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        loss = - torch.log(loss).mean()
        return loss
    
    def weighted_loss_cl_debug(self, x1, x2, frag1, frag2, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()

        batch1_pos = (batch1_ratio>0.7).type(torch.FloatTensor).cuda()
        batch2_pos = (batch2_ratio>0.7).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss1 = (pos_sim) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        loss2 = (neg1) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        loss3 = (neg2) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        
        loss1 = - torch.log(loss1)
        loss2 = - torch.log(loss2) * batch1_pos
        loss3 = - torch.log(loss3) * batch2_pos
        loss = loss1 + loss2 + loss3
        return loss.mean()
    
    def weighted_loss_cl_debug2(self, x1, x2, frag1, frag2, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()

        batch1_pos = (batch1_ratio>0.7).type(torch.FloatTensor).cuda()
        batch2_pos = (batch2_ratio>0.7).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss1 = (pos_sim) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        loss2 = (neg1) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        loss3 = (neg2) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        
        loss1 = - torch.log(loss1)
        loss2 = - torch.log(loss2) * batch1_pos
        loss3 = - torch.log(loss3) * batch2_pos
        loss = (loss1 + loss2 + loss3) / (1.0 + batch1_pos + batch2_pos)
        return loss.mean()



    def weighted_loss_cl_onlyneg(self, x1, x2, frag1, frag2, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()

        #batch1_pos = (batch1_ratio>0.7).type(torch.FloatTensor).cuda()
        #batch2_pos = (batch2_ratio>0.7).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)
        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]
        
        loss = (pos_sim) / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        
        loss = - torch.log(loss).mean()
        return loss

    def cls_loss_cl(self, x1, x2, frag1, frag2, neg, is_permuted, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)


        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)

        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]

        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)

        pred = self.classifier(neg)
        
        cls_loss = self.bce(pred, is_permuted.unsqueeze(1).float()).mean()

        #loss = - torch.log(loss).mean()
        with torch.no_grad():
            y_pred_tag = torch.round(torch.sigmoid(pred))
            #print(y_pred_tag, is_permuted)
            correct_results_sum = (y_pred_tag == is_permuted.unsqueeze(1).float()).float().sum()
            acc = correct_results_sum/is_permuted.unsqueeze(1).float().shape[0]
        return cls_loss, acc.item()


    def cls_loss_cl3(self, x1, x2, frag1, frag2, unmixed_x, mixed_x, is_permuted, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)


        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                     torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)

        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]

        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)

        pred = self.classifier(mixed_x)
        
        cls_loss = self.bce(pred, is_permuted.unsqueeze(1).float()).mean()
        #print(pred)
        #loss = - torch.log(loss).mean()
        with torch.no_grad():
            y_pred_tag = torch.round(torch.sigmoid(pred))
            #print(y_pred_tag, is_permuted)
            correct_results_sum = (y_pred_tag == is_permuted.unsqueeze(1).float()).float().sum()
            acc = correct_results_sum/is_permuted.unsqueeze(1).float().shape[0]
        return cls_loss, acc.item()

    def cls_loss_cl4(self, x1, x2, frag1, frag2, permuted_x2, is_permuted, batch1, batch2):
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        frag1_abs = frag1.norm(dim = 1)
        frag2_abs = frag2.norm(dim = 1)


        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / \
                    torch.einsum('i,j->ij', x1_abs, x2_abs)

        frag1_matrix = torch.einsum('ik,jk->ij', x1, frag1) / \
                    torch.einsum('i,j->ij', x1_abs, frag1_abs)
        frag2_matrix = torch.einsum('ik,jk->ij', x1, frag2) / \
                    torch.einsum('i,j->ij', x1_abs, frag2_abs)

        batch1_numatoms = torch.bincount(batch1)
        batch2_numatoms = torch.bincount(batch2)
        total_numatoms = batch1_numatoms + batch2_numatoms
        batch1_ratio = batch1_numatoms / total_numatoms
        batch2_ratio = batch2_numatoms / total_numatoms

        batch1_neg = (batch1_ratio<0.7).type(torch.FloatTensor).cuda()
        batch2_neg = (batch2_ratio<0.7).type(torch.FloatTensor).cuda()

        sim_matrix = torch.exp(sim_matrix / T)
        frag1_matrix = torch.exp(frag1_matrix / T)
        frag2_matrix = torch.exp(frag2_matrix / T)

        neg1 = frag1_matrix[range(batch), range(batch)]
        neg2 = frag2_matrix[range(batch), range(batch)]
        pos_sim = sim_matrix[range(batch), range(batch)]

        loss = pos_sim / (sim_matrix.sum(dim=1) + neg1 * batch1_neg + neg2 * batch2_neg - pos_sim)
        mixed_x = torch.cat([x1, permuted_x2], dim=1)
        pred = self.classifier(mixed_x)
        
        cls_loss = self.bce(pred, is_permuted.unsqueeze(1).float()).mean()
        #print(pred)
        loss = - torch.log(loss).mean()
        with torch.no_grad():
            y_pred_tag = torch.round(torch.sigmoid(pred))
            #print(y_pred_tag, is_permuted)
            correct_results_sum = (y_pred_tag == is_permuted.unsqueeze(1).float()).float().sum()
            acc = correct_results_sum/is_permuted.unsqueeze(1).float().shape[0]
        return loss + 0.1 * cls_loss, acc.item()



def train(loader, model, optimizer, device, gamma_joao):

    model.train()
    train_loss_accum = 0

    for step, (_, batch1, batch2) in enumerate(loader):
        # _, batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        # pdb.set_trace()
        x1 = model.forward_cl(batch1.x, batch1.edge_index,
                              batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index,
                              batch2.edge_attr, batch2.batch)
        loss = model.loss_cl_old(x1, x2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    # joao
    aug_prob = loader.dataset.aug_prob
    loss_aug = np.zeros(25)
    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        loader.dataset.set_augProb(_aug_prob)
        # for efficiency, we only use around 10% of data to estimate the loss
        count, count_stop = 0, len(loader.dataset) // (loader.batch_size * 10) + 1

        with torch.no_grad():
            for step, (_, batch1, batch2) in enumerate(loader):
                # _, batch1, batch2 = batch
                batch1 = batch1.to(device)
                batch2 = batch2.to(device)

                x1 = model.forward_cl(batch1.x, batch1.edge_index,
                                      batch1.edge_attr, batch1.batch)
                x2 = model.forward_cl(batch2.x, batch2.edge_index,
                                      batch2.edge_attr, batch2.batch)
                loss = model.loss_cl_old(x1, x2)
                loss_aug[n] += loss.item()
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= count

    # view selection, projected gradient descent,
    # reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
    mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b - mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b - mu, 0)
    aug_prob /= aug_prob.sum()

    return train_loss_accum / (step + 1), aug_prob


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='JOAO')
    parser.add_argument('--device', type=int, default=0, help='gpu')
    parser.add_argument('--batch_size', type=int, default=256, help='batch')
    parser.add_argument('--decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--JK', type=str, default="last",
                        choices=['last', 'sum', 'max', 'concat'],
                        help='how the node features across layers are combined.')
    parser.add_argument('--gnn_type', type=str, default="gin", help='gnn model type')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--dataset', type=str, default=None, help='root dir of dataset')
    parser.add_argument('--num_layer', type=int, default=5, help='message passing layers')
    # parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset")
    parser.add_argument('--output_model_file', type=str, default='', help='model save path')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataset loading')

    parser.add_argument('--aug_mode', type=str, default='sample')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--output_model_dir', type=str, default='')
    parser.add_argument('--old', type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if 'GEOM' in args.dataset:
        dataset = MoleculeDataset_graphcl('../datasets/{}/'.format(args.dataset), dataset=args.dataset)
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)

    # set up model
    gnn = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK,
              drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = graphcl(gnn)
    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    # print(optimizer)

    # pdb.set_trace()
    aug_prob = np.ones(25) / 25
    np.set_printoptions(precision=3, floatmode='fixed')

    for epoch in range(1, args.epochs + 1):
        print('\n\n')
        start_time = time.time()
        dataset.set_augProb(aug_prob)
        pretrain_loss, aug_prob = train(loader, model, optimizer, device, args.gamma)

        print('Epoch: {:3d}\tLoss:{:.3f}\tTime: {:.3f}\tAugmentation Probability:'.format(
            epoch, pretrain_loss, time.time() - start_time))
        print(aug_prob)

    if not args.output_model_dir == '':
        saver_dict = {'model': model.state_dict()}
        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
        torch.save(model.gnn.state_dict(), args.output_model_dir + '_model.pth')
