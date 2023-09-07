import numpy as np
import torch
import open3d as o3d
import os

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, data):

        assert self.mean is not None and self.std is not None, ('Initialize mean and std to normalize with')

        # Get mean and std
        self.mean = torch.as_tensor(self.mean, dtype=data.x.dtype, device=data.x.device)
        self.std = torch.as_tensor(self.std, dtype=data.x.dtype, device=data.x.device)

        # Apply normalization
        data.x = (data.x - self.mean)/self.std
        data.y = (data.y - self.mean)/self.std

        return data


def normal(tensor, mean, std):
    if tensor is not None:
        torch.nn.init.normal_(tensor, mean=mean, std=std)


def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor


def adjust_learning_rate(optimizer, lr_decay):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_decay


def save_model(mesh_vae, optimizer, epoch, train_loss, val_loss, checkpoint_dir):
    checkpoint = {}
    checkpoint['state_dict'] = mesh_vae.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['epoch_num'] = epoch
    checkpoint['train_loss'] = train_loss
    checkpoint['val_loss'] = val_loss
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'checkpoint_'+ str(epoch)+'.pt'))


def save_mesh(template_filepath, mesh_verts, mesh_filepath):
    mesh = o3d.io.read_triangle_mesh(template_filepath)
    vertices = np.asarray(mesh_verts)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d.io.write_triangle_mesh(mesh_filepath, mesh)