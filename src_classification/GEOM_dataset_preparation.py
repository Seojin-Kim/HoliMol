import argparse
import json
import os
import pickle
import random
from itertools import repeat
from os.path import join
import copy
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from datasets import allowable_features
from rdkit.Chem.AllChem import ReactionFromSmarts
from scipy.sparse.csgraph import connected_components
import math
from scipy.sparse.csgraph import connected_components
from rdkit.Chem import BRICS
import math

import random

def mol_fragment(mol):
    if mol is None:
        print('something wrong')
    
    Rxn = ReactionFromSmarts('[*:1]-!@[*:2]>>[*:1].[*:2]')
    fragments = Rxn.RunReactants([mol])
    reactions = []
    for (f1, f2) in fragments:
        frag1 = Chem.MolToSmiles(f1)
        frag2 = Chem.MolToSmiles(f2)
        if set([frag1, frag2]) not in reactions:
            reactions.append(set([frag1, frag2]))
    
    min_frag_size_diff = -1
    balanced_rxn = None
    #print(reactions)
    for rxn_set in reactions:
        rxn_list = list(rxn_set)
        if len(rxn_list) != 2:
            continue
        if abs(len(rxn_list[0]) - len(rxn_list[1])) < min_frag_size_diff or min_frag_size_diff < 0:
            if Chem.MolFromSmiles(rxn_list[0]) is None or Chem.MolFromSmiles(rxn_list[1]) is None:
                #print(rxn_list[0])
                #print(rxn_list[1])
                continue
            balanced_rxn = rxn_list
            min_frag_size_diff = abs(len(rxn_list[0]) - len(rxn_list[1]))
    if balanced_rxn is None:
        #print("balanced_rxn is none")
        print(Chem.MolToSmiles(mol))
        print(reactions)
        return None
    #if balanced_rxn is not None:
    #    if balanced_rxn[0].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
    #        #print("only C fragment")
    #        return None
    #    elif balanced_rxn[1].replace("C","").replace("H","").replace("(","").replace(")","").replace("[","").replace("]","") == "":
            #print("only C fragment")
    #        return None
        
    mol1 = Chem.MolFromSmiles(balanced_rxn[0])
    mol2 = Chem.MolFromSmiles(balanced_rxn[1])

    return mol1, mol2



def mol_combination(mol1, mol2):
    Rxn = ReactionFromSmarts('[*;!H0:1].[*;!H0:2]>>[*:1]-[*:2]')
    combination = Rxn.RunReactants([mol1, mol2])
    if combination is None:
        raise 'combination error'
    else:
        return combination[0][0]
    


def mol_to_graph_data_obj_simple_3D_gen(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    mol = Chem.AddHs(mol)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/

    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    #Allchem.MMFFOptimizeMolceule(mol)
    
    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data



def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    #print(Chem.MolToSmiles(mol))
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    
    
    
    brics_bond = [brics for (brics, _) in list(BRICS.FindBRICSBonds(mol))]
    
    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        

        fragmented_edges_list = []
        fragmented_edge_features_list = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            

            if (i,j) in brics_bond or (j,i) in brics_bond:
                
                continue
            
            fragmented_edges_list.append((i,j))
            fragmented_edge_features_list.append(edge_feature)
            fragmented_edges_list.append((i,j))
            fragmented_edge_features_list.append(edge_feature)


        num_atoms = len(mol.GetAtoms())
        fragmented_adj = np.zeros((num_atoms, num_atoms))
        for (i,j) in fragmented_edges_list:
            fragmented_adj[i,j] = 1
        num_components, comp_indices = connected_components(fragmented_adj)
        comp_output = torch.Tensor(comp_indices).long().unsqueeze(1)
        
        frag_indices = []
        for i in range(num_components):
            frag_indices.append(comp_indices == i)
        
        
        
        
        frags = [{} for i in range(num_components)]

        indices = [0 for i in range(num_components)]

        for (i, frag) in enumerate(frag_indices):
            for (j,val) in enumerate(frag_indices[i]):
                if val == True:
                    frags[i][j] = indices[i]
                    indices[i] += 1

        edges_lists = [[] for i in range(num_components)]

        for (i,j) in edges_list:
            for (k,frag) in enumerate(frags):
                if i in frag.keys() and j in frag.keys():
                    edges_lists[k].append((frag[i],frag[j]))


        #anchors = [None for i in range(len(brics_bond))]
        anchors = []
        for (i,j) in edges_list:
            if (i,j) in brics_bond or (j,i) in brics_bond:
                continue
            for (k,l) in brics_bond:
                if i == k and j != l:
                    if (i,j) not in anchors:
                        anchors.append((i,j))
                        anchors.append((j,i))

                elif i != k and j == l:
                    if (i,j) not in anchors:
                        anchors.append((i,j))
                        anchors.append((j,i))
        
        
        dihedrals = []
        dihedral_anchors = []
        for (i,j) in brics_bond:
            for (k1,l1) in anchors:
                for (k2,l2) in anchors:
                    if l1 == i and j == k2:
                        if atom_features_list[k1][0] == 0 or atom_features_list[l2][0] == 0:
                            continue
                        if [i,j] in dihedral_anchors:
                            continue
                        
                        dihedrals.append([k1,l1,k2,l2])
                        dihedral_anchors.append([l1,k2])
        
                

        



        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_indices = [torch.tensor(np.array(edges).T, dtype=torch.long) for edges in edges_lists]
        
        atom_features_lists = [[] for i in range(num_components)]
        for i in range(len(atom_features_list)):
            for (j,frag) in enumerate(frag_indices):
                if frag[i] == True:
                    atom_features_lists[j].append(atom_features_list[i])

        x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
        xs = [torch.tensor(np.array(atom_features), dtype=torch.long) for atom_features in atom_features_lists]

        
        edge_features_lists = [[] for i in range(num_components)]
        
        for idx, (i,j) in enumerate(edges_list):
            for (k, frag) in enumerate(frags):
                if i in frag.keys() and j in frag.keys():
                    edge_features_lists[k].append(edge_features_list[idx])
        
        
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
        edge_attrs = [torch.tensor(np.array(edge_features), dtype=torch.long) for edge_features in edge_features_lists]
        
        
    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        return []

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    
    dihedral_anchors = torch.Tensor(dihedrals).long()
    dihedral_labels = torch.zeros(15)
    dihedral_values = torch.zeros(15)
    
    dihedral_anchors_out = torch.zeros(60)
    dihedral_num = torch.Tensor([len(dihedral_anchors)]).long()
    
    
    for i in range(len(dihedrals)):
        dihedral_anchors_out[4*i:4*(i+1)] = dihedral_anchors[i]
    dihedral_anchors_out = dihedral_anchors_out.long().unsqueeze(0)
    

    for i in range(len(dihedrals)):
        p0 = np.array(positions[dihedrals[i][0]])
        p1 = np.array(positions[dihedrals[i][1]])
        p2 = np.array(positions[dihedrals[i][2]])
        p3 = np.array(positions[dihedrals[i][3]])
        b0 = -1.0*(p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1)

        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - np.dot(b0, b1)*b1
        w = b2 - np.dot(b2, b1)*b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x10 = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        dihedral_angle = math.ceil(np.degrees(np.arctan2(y, x10))/20) + 8
        
        dihedral_labels[i] = dihedral_angle
        dihedral_angle_value = np.degrees(np.arctan2(y, x10))
        dihedral_values[i] = dihedral_angle_value

    dihedral_labels = dihedral_labels.long().unsqueeze(0)
    

    positions_lists = [[] for i in range(num_components)]
    for i in range(num_atoms):
        for (j, frag) in enumerate(frag_indices):
            if frag[i] == True:
                positions_lists[j].append(positions[i])
    
    
    positions = torch.Tensor(positions)

    positions_lists = [torch.Tensor(positions_list) for positions_list in positions_lists]

    

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions, dihedral_anchors=dihedral_anchors_out, dihedral_labels=dihedral_labels, dihedral_num=dihedral_num, components=comp_output)

    datas = [Data(x=x1, edge_index=edge_index1,
                edge_attr=edge_attr1, positions=positions1) for (x1, edge_index1, edge_attr1, positions1) in zip(xs, edge_indices, edge_attrs, positions_lists)]
    
                
    #print('done?')
    #print(Chem.MolToSmiles(mol))
    #print(data['x'])
    #print(data1['x'])
    #print(data2['x'])
    
    #print(data['edge_index'])
    #print(data1['edge_index'])
    #print(data2['edge_index'])

    #print(data['edge_attr'])
    #print(data1['edge_attr'])
    #print(data2['edge_attr'])

    
    return data, datas




def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
        
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    #conformer = mol.GetConformers()[0]
    #positions = conformer.GetPositions()
    #positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=None)
    return data


def summarise():
    """ summarise the stats of molecules and conformers """
    dir_name = '{}/rdkit_folder'.format(data_folder)
    drugs_file = '{}/summary_drugs.json'.format(dir_name)

    with open(drugs_file, 'r') as f:
        drugs_summary = json.load(f)
    # expected: 304,466 molecules
    print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

    sum_list = []
    drugs_summary = list(drugs_summary.items())

    for smiles, sub_dic in tqdm(drugs_summary):
        ##### Path should match #####
        
        if sub_dic.get('pickle_path', '') == '':
            continue

        mol_path = join(dir_name, sub_dic['pickle_path'])
        with open(mol_path, 'rb') as f:
            mol_sum = {}
            mol_dic = pickle.load(f)
            conformer_list = mol_dic['conformers']
            conformer_dict = conformer_list[0]
            rdkit_mol = conformer_dict['rd_mol']
            data = mol_to_graph_data_obj_simple_3D(rdkit_mol)

            mol_sum['geom_id'] = conformer_dict['geom_id']
            mol_sum['num_edge'] = len(data.edge_attr)
            mol_sum['num_node'] = len(data.positions)
            mol_sum['num_conf'] = len(conformer_list)

            # conf['boltzmannweight'] a float for the conformer (a few rotamers)
            # conf['conformerweights'] a list of fine weights of each rotamer
            bw_ls = []
            for conf in conformer_list:
                bw_ls.append(conf['boltzmannweight'])
            mol_sum['boltzmann_weight'] = bw_ls
        sum_list.append(mol_sum)
    return sum_list


class Molecule3DDatasetFrag(InMemoryDataset):

    def __init__(self, root, n_mol, transform=None, seed=777,
                 pre_transform=None, pre_filter=None, empty=False, **kwargs):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)
        if 'smiles_copy_from_3D_file' in kwargs:  # for 2D Datasets (SMILES)
            self.smiles_copy_from_3D_file = kwargs['smiles_copy_from_3D_file']
        else:
            self.smiles_copy_from_3D_file = None

        self.root, self.seed = root, seed
        self.n_mol = n_mol
        self.pre_transform, self.pre_filter = pre_transform, pre_filter

        super(Molecule3DDatasetFrag, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.data1, self.slices1 = torch.load(self.processed_paths[0] + "_1")
            self.data2, self.slices2 = torch.load(self.processed_paths[0] + "_2")

        
        print('root: {},\ndata: {},\nn_mol: {},\n'.format(
            self.root, self.data, self.n_mol))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]

            item1, slices1 = self.data1[key], self.slices1[key]
            s = list(repeat(slice(None), item.dim()))
            s[data1.__cat_dim__(key, item1)] = slice(slices1[idx], slices1[idx+1])
            data1[key] = item1[s]

            item2, slices2 = self.data2[key], self.slices2[key]
            s = list(repeat(slice(None), item.dim()))
            s[data2.__cat_dim__(key, item2)] = slice(slices2[idx], slices2[idx+1])
            data2[key] = item2[s]
        return data, data1, data2

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []
        data1_list = []
        data2_list = []
        data_smiles_list = []

        downstream_task_list = ["tox21", "toxcast", "clintox", "bbbp", "sider", "muv", "hiv", "bace", "esol", "lipophilicity"]
        whole_SMILES_set = set()
        '''
        for task in downstream_task_list:
            print("====== {} ======".format(task))
            file_path = "../datasets/molecule_datasets/{}/processed/smiles.csv".format(task)
            SMILES_list = load_SMILES_list(file_path)
            temp_SMILES_set = set(SMILES_list)
            whole_SMILES_set = whole_SMILES_set | temp_SMILES_set
        print("len of downstream SMILES:", len(whole_SMILES_set))
        '''
        if self.smiles_copy_from_3D_file is None:  # 3D datasets
            print('something wrong')
        else:  # 2D datasets
            with open(self.smiles_copy_from_3D_file, 'r') as f:
                lines = f.readlines()
            for smiles in lines:
                data_smiles_list.append(smiles.strip())
            data_smiles_list = list(dict.fromkeys(data_smiles_list))

            # load 3D structure
            dir_name = '{}/rdkit_folder'.format(data_folder)
            drugs_file = '{}/summary_drugs.json'.format(dir_name)
            with open(drugs_file, 'r') as f:
                drugs_summary = json.load(f)
            # expected: 304,466 molecules
            print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

            mol_idx, idx, notfound = 0, 0, 0
            err_cnt = 0
            one_functional_group_list = []
            total_cc = 0
            for smiles in tqdm(data_smiles_list):
                sub_dic = drugs_summary[smiles]
                if "pickle_path" not in sub_dic.keys():
                    print('weired')
                    continue
                
                mol_path = join(dir_name, sub_dic['pickle_path'])
                with open(mol_path, 'rb') as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic['conformers']
                    new_list = sorted(conformer_list, key=lambda d: d['relativeenergy']) 


                    conformer = new_list[0]
                    rdkit_mol = conformer['rd_mol']
                    
                    #data = mol_to_graph_data(rdkit_mol)
                    
                    data, datas = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    
                    # print(Chem.MolToSmiles(conformer['rd_mol']))
                    # print(data.x)
                    # print(data1.x)
                    # print(data1.edge_attr)
                    # print(data1.edge_index.T)
                    # print(data2.x)
                    # print(data2.edge_attr)
                    # print(data2.edge_index.T)

                    # print(data.positions)
                    # print(data1.positions)
                    # print(data2.positions)
                    # raise dd
                    #rdkit_mol = Chem.MolFromSmiles(Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(conformer['rd_mol']))))
                    #mols = mol_fragment(rdkit_mol)
                    #fp = Chem.RDKFingerprint(rdkit_mol)
                    #if mols is None:
                    #    err_cnt += 1
                    #    continue      
                        #if len(one_functional_group_list) == 2:
                        #    mol_combination(one_functional_group_list[0], one_functional_group_list[1])
                    #else:
                    #    if mols[0] is None or mols[1] is None:
                    #        print(mols[0], mols[1])
                    #        continue
                    #    try:
                    #        data1 = mol_to_graph_data_obj_simple_3D_gen(mols[0])
                    #        data2 = mol_to_graph_data_obj_simple_3D_gen(mols[1])
                    #    except:
                    #        print('exception')
                    #        continue

                    data.mol_id = torch.tensor([mol_idx])
                    data.id = torch.tensor([idx])
                    data_list.append(data)
                    
                    datas, slices = self.collate(datas)
                    torch.save((datas, slices), self.processed_paths[0] + "_" + str(idx))

                    mol_idx += 1
                    idx += 1

                    
        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)
    
        data, slices = self.collate(data_list)
        
        
        torch.save((data, slices), self.processed_paths[0])
        
        
        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers have been processed" % idx)
        return


def load_SMILES_list(file_path):
    SMILES_list = []
    with open(file_path, 'rb') as f:
        for line in tqdm(f.readlines()):
            SMILES_list.append(line.strip().decode())
    return SMILES_list


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum', type=bool, default=False, help='cal dataset stats')
    parser.add_argument('--n_mol', type=int, help='number of unique smiles/molecules')
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--data_path', type=str)
    args = parser.parse_args()

    data_folder = args.data_folder

    if args.sum:
        sum_list = summarise()
        with open('{}/summarise.json'.format(data_folder), 'w') as fout:
            json.dump(sum_list, fout)

    else:
        n_mol = args.n_mol
        root_2d = '{}/GEOM_2D_3D_nmol{}_brics'.format(data_folder, n_mol)

        # Generate 3D Datasets (2D SMILES + 3D Conformer)
        #Molecule3DDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper)
        # Generate 2D Datasets (2D SMILES)
        Molecule3DDatasetFrag(root=root_2d, n_mol=n_mol,
                          smiles_copy_from_3D_file=args.data_path)
    
    ##### to data copy to SLURM_TMPDIR under the `datasets` folder #####
    '''
    wget https://dataverse.harvard.edu/api/access/datafile/4327252
    mv 4327252 rdkit_folder.tar.gz
    cp rdkit_folder.tar.gz $SLURM_TMPDIR
    cd $SLURM_TMPDIR
    tar -xvf rdkit_folder.tar.gz
    '''

