import argparse
import glob
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from psbody.mesh import Mesh
from model.mesh_operations import get_vert_connectivity
from helpers import Normalize


class MeshDataset(InMemoryDataset):
    def __init__(self, root_dir, dtype='train', transform=None, pre_transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.pre_tranform = pre_transform
        self.data_file = glob.glob(self.root_dir + '/*.ply')

        super(MeshDataset, self).__init__(root_dir, transform, pre_transform)
        if dtype == 'train':
            data_path = self.processed_paths[0]
        elif dtype == 'val':
            data_path = self.processed_paths[1]
        elif dtype == 'test':
            data_path = self.processed_paths[2]
        else:
            raise Exception("train, val and test are supported data types")

        norm_path = self.processed_paths[3]
        norm_dict = torch.load(norm_path)
        self.mean, self.std = norm_dict['mean'], norm_dict['std']

        self.data, self.slices = torch.load(data_path)
        if self.transform:
            self.data = [self.transform(td) for td in self.data]

    @property
    def raw_file_names(self):
        return self.data_file

    @property
    def processed_file_names(self):
        processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt']
        return processed_files

    def process(self):

        dataset_size = len(self.data_file)
        train_data, val_data, test_data = [], [], []
        train_vertices = []
        for idx, data_file in tqdm(enumerate(self.data_file)):

            # Read mesh and connectivity info
            mesh = Mesh(filename=data_file)
            mesh_verts = torch.Tensor(mesh.v)
            adjacency = get_vert_connectivity(mesh.v, mesh.f).tocoo()
            edge_index = torch.Tensor(np.vstack((adjacency.row, adjacency.col)))
            data = Data(x=mesh_verts, y=mesh_verts, edge_index=edge_index)

            # Add current mesh to specific dataset split
            if idx <= 0.25*dataset_size:
                test_data.append(data)
            elif idx <= 0.30*dataset_size:
                val_data.append(data)
            else:
                train_data.append(data)
                train_vertices.append(mesh.v)

        # Get mean and std for normalization
        mean_train = torch.Tensor(np.mean(train_vertices, axis=0)) + 0.000001
        std_train = torch.Tensor(np.std(train_vertices, axis=0)) + 0.000001
        norm_dict = {'mean': mean_train, 'std': std_train}
        if self.pre_transform is not None:
            if hasattr(self.pre_transform, 'mean') and hasattr(self.pre_transform, 'std'):
                if self.pre_tranform.mean is None:
                    self.pre_tranform.mean = mean_train
                if self.pre_transform.std is None:
                    self.pre_tranform.std = std_train
            train_data = [self.pre_transform(td) for td in train_data]
            val_data = [self.pre_transform(td) for td in val_data]
            test_data = [self.pre_transform(td) for td in test_data]

        torch.save(self.collate(train_data), self.processed_paths[0])
        torch.save(self.collate(val_data), self.processed_paths[1])
        torch.save(self.collate(test_data), self.processed_paths[2])
        torch.save(norm_dict, self.processed_paths[3])

def prepare_dataset(path):
    MeshDataset(path, pre_transform=Normalize())



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Data preparation for Variational Mesh Autoencoders')
    parser.add_argument('-d', '--data_dir', help='path where data is saved')
    args = parser.parse_args()
    data_dir = args.data_dir

    prepare_dataset(data_dir)


