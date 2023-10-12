import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import args
from models import GNN, SchNet
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, GlobalAttention
from tqdm import tqdm
from util import dual_CL, dual_Frag_CL, node_CL, node_filip, node_sim
from pretrain_JOAO import graphcl, graphcl3d
from datasets import Molecule3DDatasetFragRandomaug3d_2
from torch_scatter import scatter

def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(gnn.state_dict(), args.output_model_dir + '_model.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
            
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')

        else:
            torch.save(gnn.state_dict(), args.output_model_dir + '_model_final.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
             
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_final.pth')
    return


def train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, epoch):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    if molecule_projection_layer is not None:
        molecule_projection_layer.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum1 ,CL_acc_accum2 = 0, 0, 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for step, (batch, batch1, orig_batch, orig_batch1, idx) in enumerate(l):
        batch = batch.to(device)
        batch1 = batch1.to(device)
        orig_batch.to(device)
        orig_batch1.to(device)

        frag_batch = [torch.tensor([i] * idx[i]) for i in range(len(idx))]
        frag_batch = torch.cat(frag_batch).to(device)
        molecule_2D_repr_orig,_,_ = molecule_model_2D.forward_cl_node(orig_batch.x, orig_batch.edge_index, orig_batch.edge_attr, orig_batch.batch)
        molecule_2D_repr = molecule_model_2D.forward_cl(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        frag_2D_repr = molecule_model_2D.forward_pool(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        molecule_2D_repr_mixed = molecule_model_2D.attention(frag_2D_repr, frag_batch)
        molecule_2D_repr_mixed = molecule_model_2D.forward_project(molecule_2D_repr_mixed)
        
        loss_2D = molecule_model_2D.loss_cl_graphcl(molecule_2D_repr, molecule_2D_repr_mixed)    

        if args.model_3d == 'schnet' or args.model_3d == 'spherenet':
            molecule_3D_repr_orig, molecule_3D_repr = molecule_model_3D.forward_cl_node(orig_batch.x[:, 0], orig_batch.positions + (torch.randn_like(orig_batch.positions).cuda()), orig_batch.batch) 
            
            frag_3D_repr = molecule_model_3D.forward_pool(orig_batch1.x[:, 0], orig_batch1.positions + (torch.randn_like(orig_batch1.positions).cuda()), orig_batch1.batch)
            
            molecule_3D_repr_mixed = molecule_model_3D.attention(frag_3D_repr, frag_batch)
            
            molecule_3D_repr_mixed = molecule_model_3D.forward_project(molecule_3D_repr_mixed)
            
            loss_3D = molecule_model_3D.loss_cl_graphcl(molecule_3D_repr, molecule_3D_repr_mixed)   
            
        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)
        
        binc = torch.bincount(orig_batch.batch)

        batch_size = len(binc) 
        offset = torch.cat([torch.Tensor([0]).long().cuda(), torch.cumsum(binc, dim=0)], dim=0)
        
        frag_binc = torch.bincount(frag_batch)
        frag_offset = torch.cat([torch.Tensor([0]).long().cuda(), torch.cumsum(frag_binc, dim=0)], dim=0) 
        node_idx_all = torch.tensor([]).cuda()
        dihedral_labels_all = torch.tensor([]).cuda()
        
        component_batch = orig_batch.components[:,0]

        for b in range(batch_size):
            
            b_offset = offset[b]
            num_dihedral = orig_batch.dihedral_num[b]
            
            component_batch[offset[b]:offset[b+1]] += frag_offset[b]
            
            if num_dihedral == 0:
                continue
                
            node_idx = orig_batch.dihedral_anchors[b,:4*num_dihedral].clone()
            dihedral_labels = orig_batch.dihedral_labels[b,:num_dihedral].clone()
            
            node_idx += b_offset
            
            node_idx_all = torch.cat([node_idx_all, node_idx])
            dihedral_labels_all = torch.cat([dihedral_labels_all, dihedral_labels])
        
        frag_2d = molecule_model_2D.average_pool(molecule_2D_repr_orig, component_batch)
        frag_3d = molecule_model_3D.average_pool(molecule_3D_repr_orig, component_batch)
        
        frag_2d = molecule_model_2D.forward_project(frag_2d)
        frag_3d = molecule_model_3D.forward_project(frag_3d)
        
        frag_CL_loss, frag_CL_acc = node_filip(frag_2d, frag_3d, frag_batch, args)
            
        anchor = torch.index_select(molecule_2D_repr_orig, 0, node_idx_all.long()).reshape(-1, 4, 300)
        anchor_reverse = torch.zeros_like(anchor).cuda()
        anchor_reverse[:,0,:] = anchor[:,3,:]
        anchor_reverse[:,1,:] = anchor[:,2,:]
        anchor_reverse[:,2,:] = anchor[:,1,:]
        anchor_reverse[:,3,:] = anchor[:,0,:]
        
        anchor = anchor.reshape(anchor.shape[0], -1)
        anchor_out = molecule_model_2D.forward_aux2(anchor)
        loss_anchor1 = molecule_model_2D.ce2(anchor_out, dihedral_labels_all.long())
        
        anchor_reverse = anchor_reverse.reshape(anchor.shape[0], -1)
        anchor_out2 = molecule_model_2D.forward_aux2(anchor_reverse)
        loss_anchor2 = molecule_model_2D.ce2(anchor_out2,dihedral_labels_all.long())
        
        loss_anchor = (loss_anchor1 + loss_anchor2)/2
        correct1 = (anchor_out.argmax(dim=1) == dihedral_labels_all.long()).sum()
        correct2 = (anchor_out2.argmax(dim=1) == dihedral_labels_all.long()).sum()
        
        acc = (correct1 + correct2) / (2 * orig_batch.dihedral_num.sum())
        AE_loss_accum += loss_anchor.detach().cpu().item()
        AE_acc_accum += acc
        
        CL_loss_accum += (CL_loss.detach().cpu().item() + frag_CL_loss.detach().cpu().item())/2 #CL_loss.d
        CL_acc_accum1 += CL_acc
        CL_acc_accum2 += frag_CL_acc

        loss = 0
        if epoch < 5:
            loss += loss_2D + loss_3D #+ loss_anchor 
        else:
            loss += loss_2D + (CL_loss+frag_CL_loss)/2 + loss_3D #+ loss_anchor 
        
        
        #if args.alpha_2 > 0:
        #    loss += AE_loss * args.alpha_2
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(molecule_model_2D.parameters(), 5)
        torch.nn.utils.clip_grad_norm_(molecule_model_3D.parameters(), 5)
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum1 /= len(loader)
    CL_acc_accum2 /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    temp_loss = CL_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print('CL Loss: {:.5f}\tCL graph Acc: {:.5f}\tCL local Acc: {:.5f}\tAE Loss: {:.5f}\tAE Acc: {:.5f}\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum1, CL_acc_accum2, AE_loss_accum, AE_acc_accum, time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        n_mol = args.n_mol
        root_2d = '{}/GEOM_2D_3D_nmol50000_brics_manual_0111_dihedral_index'.format(args.data_folder)
        dataset = Molecule3DDatasetFragRandomaug3d_2(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    if 'QM9' in args.dataset:
        n_mol = args.n_mol
        root_2d = '{}/QM9_2D_3D_cut_singlebond_manual'.format(args.data_folder, args.n_mol)
        dataset = Molecule3DDatasetFragRandomaug3d(root=root_2d, n_mol=n_mol, choose=args.choose,
                          smiles_copy_from_3D_file='%s/processed/smiles.csv' % root_2d)
        dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    aug_prob = np.ones(25) / 25
    dataset.set_augProb(aug_prob)
    

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    molecule_model_2D = graphcl(gnn).to(device)
    

    print('Using 3d model\t', args.model_3d)
    molecule_projection_layer = None
    if args.model_3d == 'schnet':
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout)
        molecule_model_3D = graphcl3d(molecule_model_3D).to(device)
    elif args.model_3d == 'spherenet':
        #molecule_model_3D = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
        #          hidden_channels=128, out_channels=1, int_emb_size=64,
        #          basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
        #          num_spherical=3, num_radial=6, envelope_exponent=5,
        #          num_before_skip=1, num_after_skip=2, num_output_layers=3)   
        molecule_model_3D = SphereNet(energy_and_force=False, cutoff=5.0, num_layers=4,
                  hidden_channels=128, out_channels=300, int_emb_size=64,
                  basis_emb_size_dist=8, basis_emb_size_angle=8, basis_emb_size_torsion=8, out_emb_channels=256,
                  num_spherical=3, num_radial=6, envelope_exponent=5,
                  num_before_skip=1, num_after_skip=2, num_output_layers=3)   
        molecule_model_3D = graphcl3d(molecule_model_3D).to(device)
    else:
        raise NotImplementedError('Model {} not included.'.format(args.model_3d))

    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    if molecule_projection_layer is not None:
        model_param_group.append({'params': molecule_projection_layer.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10
    epoch = 0
    for epoch in range(1, args.epochs + 1):
    #while epoch < 100:
        print('epoch: {}'.format(epoch))
        
        train(args, molecule_model_2D, molecule_model_3D, device, loader, optimizer, epoch) ###
    #        epoch += 1
        
            

    save_model(save_best=False)
