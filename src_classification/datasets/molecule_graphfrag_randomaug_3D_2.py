
import os
from itertools import repeat

import numpy as np
import torch
from torch_geometric.utils import subgraph, to_networkx
from torch_geometric.data import Data, InMemoryDataset, Batch

from .molecule_datasets import MoleculeDataset
from os.path import join
import copy

def mol_fragment(mol):
    if mol is None:
        print('something wrong')
    Rxn = ReactionFromSmarts('[C:1]-!@[C:2]>>[C:1].[C:2]')
    fragments = Rxn.RunReactants([mol])
    reactions = []
    for (f1, f2) in fragments:
        frag1 = Chem.MolToSmiles(f1)
        frag2 = Chem.MolToSmiles(f2)
        if set([frag1, frag2]) not in reactions:
            reactions.append(set([frag1, frag2]))
    
    min_frag_size_diff = -1
    balanced_rxn = None

    for rxn_set in reactions:
        rxn_list = list(rxn_set)
        if len(rxn_list) != 2:
            continue
        if abs(len(rxn_list[0]) - len(rxn_list[1])) < min_frag_size_diff or min_frag_size_diff < 0:
            balanced_rxn = rxn_list
            min_frag_size_diff = abs(len(rxn_list[0]) - len(rxn_list[1]))
    
    if balanced_rxn is not None:
        mol1 = Chem.MolFromSmiles(balanced_rxn[0])
        mol2 = Chem.MolFromSmiles(balanced_rxn[1])

        return mol1, mol2

def mol_to_graph_data_obj_simple_3D(mol):
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
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, positions=positions)
    return data




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


class Molecule3DDatasetFragRandomaug3d_2(MoleculeDataset):

    def __init__(self, root, n_mol,choose = 0,transform=None, seed=777,
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
        self.aug_prob = None
        self.aug_mode = 'no_aug'
        self.aug_strength = 0.2
        self.choose = choose
        self.choosetwo_idx = [[0,1], [0,2],[0,3], [1,2], [1,3], [2,3]]
        self.choosethree_idx = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        print("I choose")
        print(choose)
        self.augmentations = [self.node_drop, self.subgraph,
                              self.edge_pert, self.attr_mask, lambda x: x]
        super(Molecule3DDatasetFragRandomaug3d_2, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
#             for i in range(50000):
#                 brics_data = torch.load(self.processed_paths[0] + "_" + str(i))
#                 if brics_data[1] == None:
#                     self.brics.append(brics_data[0])
#                     self.brics_slice.append(dict({'x': None, 'edge_index': None, 'edge_attr': None}))
#                 else:
#                     self.brics.append(brics_data[0])
#                     self.brics_slice.append(brics_data[1])

        
        print('root: {},\ndata: {},\nn_mol: {},\n'.format(
            self.root, self.data, self.n_mol))


    def set_augMode(self, aug_mode):
        self.aug_mode = aug_mode

    def set_augStrength(self, aug_strength):
        self.aug_strength = aug_strength

    def set_augProb(self, aug_prob):
        self.aug_prob = aug_prob

    def node_drop(self, data):
        #print(data.x.size())
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * self.aug_strength)

        idx_perm = np.random.permutation(node_num)
        idx_nodrop = idx_perm[drop_num:].tolist()
        idx_nodrop.sort()

        edge_idx, edge_attr = subgraph(subset=idx_nodrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_attr,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nodrop]
        data.__num_nodes__, _ = data.x.shape
        return data

    def edge_pert(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * self.aug_strength)

        # delete edges
        idx_drop = np.random.choice(edge_num, (edge_num - pert_num),
                                    replace=False)
        edge_index = data.edge_index[:, idx_drop]
        edge_attr = data.edge_attr[idx_drop]

        # add edges
        adj = torch.ones((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 0
        # edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        edge_index_nonexist = torch.nonzero(adj, as_tuple=False).t()
        idx_add = np.random.choice(edge_index_nonexist.shape[1],
                                   pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]
        # random 4-class & 3-class edge_attr for 1st & 2nd dimension
        edge_attr_add_1 = torch.tensor(np.random.randint(
            4, size=(edge_index_add.shape[1], 1)))
        edge_attr_add_2 = torch.tensor(np.random.randint(
            3, size=(edge_index_add.shape[1], 1)))
        edge_attr_add = torch.cat((edge_attr_add_1, edge_attr_add_2), dim=1)
        edge_index = torch.cat((edge_index, edge_index_add), dim=1)
        edge_attr = torch.cat((edge_attr, edge_attr_add), dim=0)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data

    def edge_del(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * self.aug_strength)

        # delete edges
        idx_drop = np.random.choice(edge_num, (edge_num - pert_num),
                                    replace=False)
        edge_index = data.edge_index[:, idx_drop]
        edge_attr = data.edge_attr[idx_drop]

        # add edges
        #adj = torch.ones((node_num, node_num))
        #adj[edge_index[0], edge_index[1]] = 0
        # edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        #edge_index_nonexist = torch.nonzero(adj, as_tuple=False).t()
        #idx_add = np.random.choice(edge_index_nonexist.shape[1],
        #                           pert_num, replace=False)
        #edge_index_add = edge_index_nonexist[:, idx_add]
        # random 4-class & 3-class edge_attr for 1st & 2nd dimension
        #edge_attr_add_1 = torch.tensor(np.random.randint(
        #    4, size=(edge_index_add.shape[1], 1)))
        #edge_attr_add_2 = torch.tensor(np.random.randint(
        #    3, size=(edge_index_add.shape[1], 1)))
        #edge_attr_add = torch.cat((edge_attr_add_1, edge_attr_add_2), dim=1)
        #edge_index = torch.cat((edge_index, edge_index_add), dim=1)
        #edge_attr = torch.cat((edge_attr, edge_attr_add), dim=0)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data


    def attr_mask(self, data):

        _x = data.x.clone()
        node_num, _ = data.x.size()
        mask_num = int(node_num * self.aug_strength)

        token = data.x.float().mean(dim=0).long()
        idx_mask = np.random.choice(
            node_num, mask_num, replace=False)

        _x[idx_mask] = token
        data.x = _x
        return data

    def subgraph(self, data):

        G = to_networkx(data)
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1 - self.aug_strength))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_idx, edge_attr = subgraph(subset=idx_nondrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_attr,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.__num_nodes__, _ = data.x.shape
        return data


    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]

        load_brics = torch.load(self.processed_paths[0]+"_"+str(idx))

        if load_brics[1] == None:
            brics_datas = [Data()]
            for key in load_brics[0].keys:
                item, slices = self.data[key], self.slices[key]
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
                brics_datas[0][key] = item[s] 

        else:
            brics_datas = [Data() for _ in range(len(load_brics[1]['x'])-1)]
            for i in range(len(load_brics[1]['x'])-1):
                for key in load_brics[0].keys:
                    item, slices = load_brics[0][key], load_brics[1][key]
                    s = list(repeat(slice(None), item.dim()))
                    s[data.__cat_dim__(key, item)] = slice(slices[i], slices[i+1])
                    brics_datas[i][key] = item[s]
                    
        if self.aug_mode == 'no_aug':
            n_aug, n_aug1, n_aug2 = 4, 4, 4
            data = self.augmentations[n_aug](data)
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'choose':
            two_augmentations = [self.augmentations[self.choose], lambda x: x]
            n_aug = np.random.choice(2,1)[0]
            n_aug1 = np.random.choice(2,1)[0]
            n_aug2 = np.random.choice(2,1)[0]
            data = self.augmentations[n_aug](data)
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'edgedel':
            two_augmentations = [self.edge_del, lambda x: x]
            n_aug = np.random.choice(2,1)[0]
            n_aug1 = np.random.choice(2,1)[0]
            n_aug2 = np.random.choice(2,1)[0]
            data = self.augmentations[n_aug](data)
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'choosetwo':
            if self.choose == 0:
                two_augmentations = [self.augmentations[3], self.augmentations[0], lambda x:x]
            elif self.choose == 1:
                two_augmentations = [self.augmentations[1], self.augmentations[2], lambda x:x]
                
            n_aug = np.random.choice(3,1)[0]
            aug_data = two_augmentations[n_aug](data.clone())
            aug_brics_datas = copy.deepcopy(brics_datas)
            for i in range(len(brics_datas)):
                n_aug = np.random.choice(3,1)[0]
                aug_brics_datas[i] = two_augmentations[n_aug](aug_brics_datas[i].clone())
            
            return aug_data, Batch.from_data_list(aug_brics_datas), data, Batch.from_data_list(brics_datas), len(brics_datas)

        elif self.aug_mode == 'choosethree':
            three_augmentations = [self.augmentations[self.choosethree_idx[self.choose][0]], self.augmentations[self.choosethree_idx[self.choose][1]], self.augmentations[self.choosethree_idx[self.choose][2]]]
            
            n_aug = np.random.choice(3,1)[0]
            n_aug1 = np.random.choice(3,1)[0]
            n_aug2 = np.random.choice(3,1)[0]
            data = three_augmentations[n_aug](data)
            data1 = three_augmentations[n_aug1](data1)
            data2 = three_augmentations[n_aug2](data2)
        elif self.aug_mode == 'uniform':
            n_aug_init = np.random.choice(25, 1)[0]
            n_aug1, n_aug2 = n_aug_init // 5, n_aug_init % 5
            n_aug = np.random.choice(5,1)[0]
            data = self.augmentations[n_aug](data)
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'sample':
            n_aug_init = np.random.choice(25, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 =  n_aug_init // 5, n_aug_init % 5
            n_aug = np.random.choice(5,1)[0]
            data = self.augmentations[n_aug](data)
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        else:
            raise ValueError
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
            for smiles in tqdm(data_smiles_list):
                sub_dic = drugs_summary[smiles]
                mol_path = join(dir_name, sub_dic['pickle_path'])
                with open(mol_path, 'rb') as f:
                    mol_dic = pickle.load(f)
                    conformer_list = mol_dic['conformers']
                    conformer = conformer_list[0]
                    rdkit_mol = conformer['rd_mol']

                    mols = mol_fragment(rdkit_mol)
                    
                    data = mol_to_graph_data_obj_simple_3D(rdkit_mol)
                    
                    if mols is None:
                        err_cnt += 1
                        data1 = data
                        data2 = data
                    else:
                        data1 = mol_to_graph_data(mols[0])
                        data2 = mol_to_graph_data(mols[1])

                    data.mol_id = torch.tensor([mol_idx])
                    data.id = torch.tensor([idx])
                    data_list.append(data)
                    data1.mol_id = torch.tensor([mol_idx])
                    data1.id = torch.tensor([idx])
                    data1_list.append(data1)
                    data2.mol_id = torch.tensor([mol_idx])
                    data2.id = torch.tensor([idx])
                    data2_list.append(data2)

                    mol_idx += 1
                    idx += 1
            print(err_cnt)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
    
        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False)
    
        data, slices = self.collate(data_list)
        data1, slices1 = self.collate(data1_list)
        data2, slices2 = self.collate(data2_list)
        torch.save((data, slices), self.processed_paths[0])
        torch.save((data1, slices1), self.processed_paths[0] + "_1")
        torch.save((data2, slices2), self.processed_paths[0] + "_2")

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
