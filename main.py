import argparse
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import time
from psbody.mesh import Mesh
import model.mesh_operations as mesh_ops
from input.config_parser import read_config
from input.data import MeshDataset
from model.mesh_vae import MeshVAE
from helpers import Normalize, scipy_to_torch_sparse, adjust_learning_rate, save_model, save_mesh


def main(args):

    #
    # I/O
    #

    # Read config file
    if not os.path.exists(args.conf):
        print('Config not found' + args.conf)
    config = read_config(args.conf)

    # Read template mesh
    template_filepath = config['template_filepath']
    template_mesh = Mesh(filename=template_filepath)

    # Set checkpoint dir
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Set output dirs
    visualize = config['visualize']
    output_dir = config['output_dir']
    if visualize is True and not output_dir:
        print('No output directory is provided. Checkpoint directory will be used to store the output meshes')
        output_dir = checkpoint_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get other config settings
    eval_flag = config['eval']
    lr = config['learning_rate']
    lr_decay = config['learning_rate_decay']
    weight_decay = config['weight_decay']
    total_epochs = config['epoch']
    workers_thread = config['workers_thread']
    opt = config['optimizer']
    batch_size = config['batch_size']
    val_losses = []

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #
    # Calculate transforms for mesh sampling
    #

    print('Generating transforms')
    M, A, D, U = mesh_ops.generate_transform_matrices(template_mesh, config['downsampling_factors'])

    D_t = [scipy_to_torch_sparse(d).to(device) for d in D]
    U_t = [scipy_to_torch_sparse(u).to(device) for u in U]
    A_t = [scipy_to_torch_sparse(a).to(device) for a in A]
    num_nodes = [len(M[i].v) for i in range(len(M))]


    #
    # Load dataset
    #

    print('Loading dataset')
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config['data_dir']

    normalize_transform = Normalize()
    dataset = MeshDataset(data_dir, dtype='train', pre_transform=normalize_transform)
    dataset_test = MeshDataset(data_dir, dtype='test', pre_transform=normalize_transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers_thread)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=workers_thread)


    #
    # Load model
    #

    print('Loading model')

    # Get model and set optimizer
    start_epoch = 1
    mesh_vae = MeshVAE(dataset, config, D_t, U_t, A_t, num_nodes)
    if opt == 'adam':
        optimizer = torch.optim.Adam(mesh_vae.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(mesh_vae.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise Exception('No optimizer provided')

    # Load model checkpoint if set
    checkpoint_file = config['checkpoint_file']
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch_num']
        mesh_vae.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    mesh_vae.to(device)


    #
    # Eval pre-trained model if set
    #

    if eval_flag:
        val_loss = evaluate(mesh_vae, test_loader, dataset_test, template_filepath, output_dir, device, visualize)
        print('val loss', val_loss)
        return


    #
    # Start network training
    #

    # Set training schedule for loss weighting parameter beta
    beta_values = [0.0001, 0.00025, 0.0005, 0.00075, 0.001]
    beta_epoch_changes = [50, 100, 150, 200]
    beta = beta_values[0]

    # Iterate over epochs
    best_val_loss = float('inf')
    val_loss_history = []
    for epoch in range(start_epoch, total_epochs + 1):

        # Adjust beta based on current epoch
        if epoch == beta_epoch_changes[0]:
            beta = beta_values[1]
        elif epoch == beta_epoch_changes[1]:
            beta = beta_values[2]
        elif epoch == beta_epoch_changes[2]:
            beta = beta_values[3]
        elif epoch == beta_epoch_changes[3]:
            beta = beta_values[4]

        # Train network
        epoch_start = time.time()
        print("Training for epoch ", epoch)
        train_loss_total, train_loss, train_loss_l1, train_loss_kl = train(mesh_vae, train_loader, len(dataset), optimizer, device, beta)
        print('Epoch ', epoch,', train loss ', train_loss, ', beta ', beta)
        print(' Train L1 loss ', train_loss_l1)
        print(' Train KL loss ', train_loss_kl)
        epoch_end = time.time()
        print('Epoch execution time: {}'.format(epoch_end-epoch_start))


        # Eval current model checkpoint
        if epoch % 25 == 0:

            val_loss = evaluate(mesh_vae, test_loader, dataset_test, template_filepath, output_dir, device, visualize=visualize)
            print('Epoch ', epoch,', train loss ', train_loss, ', val loss ', val_loss)
            save_model(mesh_vae, optimizer, epoch, train_loss, val_loss, checkpoint_dir)
            best_val_loss = val_loss
            val_loss_history.append(val_loss)
            val_losses.append(best_val_loss)

        if opt=='sgd':
            adjust_learning_rate(optimizer, lr_decay)

    if torch.cuda.is_available():
        torch.cuda.synchronize()



def train(mesh_vae, train_loader, len_dataset, optimizer, device, beta):

    # Switch to train mode
    mesh_vae.train()

    # Iterate through train data
    total_loss = 0
    for data in train_loader:

        # Move to device
        data = data.to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Pass through network
        reconstruction, mu, log_var, z = mesh_vae(data)

        # Calculate loss
        loss_l1 = F.mse_loss(reconstruction, data.y)
        loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1)
        loss_kl = torch.mean(loss_kl)
        loss = loss_l1 + beta*loss_kl
        total_loss += loss.item()

        # Backprop
        loss.backward()
        optimizer.step()

    return total_loss / len_dataset, loss, loss_l1, loss_kl



def evaluate(mesh_vae, test_loader, dataset, template_filepath, output_dir, device, visualize=False):

    # Switch to eval mode
    mesh_vae.eval()

    # Iterate through eval data
    total_loss = 0
    for i, data in enumerate(test_loader):

        # Move to device
        data = data.to(device)

        # Pass through network
        with torch.no_grad():
            out, _, _, _ = mesh_vae(data)

        # Calculate loss
        loss = F.l1_loss(out, data.y)
        total_loss += data.num_graphs * loss.item()

        # If set, save ground truth and predicted meshes
        if visualize:

            print("Saving output")

            # Detach and denormalize data
            save_out = out.detach().cpu().numpy()
            save_out = save_out*dataset.std.numpy()+dataset.mean.numpy()
            expected_out = (data.y.detach().cpu().numpy())*dataset.std.numpy()+dataset.mean.numpy()
            pred = (out.detach())*dataset.std+dataset.mean
            gt = (data.y.detach())*dataset.std+dataset.mean

            # Calculate eval loss on meshes
            eval_l1_loss = F.l1_loss(pred, gt)
            print("Current eval L1 loss: {}".format(eval_l1_loss))

            # Set output filepaths
            pred_mesh_filepath = os.path.join(output_dir, "case_" + str(i) + "_pred.ply")
            gt_mesh_filepath = os.path.join(output_dir, "case_" + str(i) + "_gt.ply")

            # Save predicted and ground truth meshes
            save_mesh(template_filepath, save_out, pred_mesh_filepath)
            save_mesh(template_filepath, expected_out, gt_mesh_filepath)


    return total_loss/len(dataset)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch Trainer')
    parser.add_argument('-c', '--conf', help='path of config file')
    parser.add_argument('-d', '--data_dir', help='path where the data is stored')
    parser.add_argument('-cp', '--checkpoint_dir', help='path where the checkpoints are stored')

    args = parser.parse_args()

    if args.conf is None:
        args.conf = os.path.join(os.path.dirname(__file__), 'settings.cfg')
        print('Configuration file not specified, trying to load '
              'it from current directory', args.conf)

    main(args)
