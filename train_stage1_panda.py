# traing the vector quantizer.

from genericpath import isfile
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import MultivariateNormal

from einops import rearrange

import torch.optim as t_optim
import json

from os import path as osp

from tqdm import tqdm

from modules.quantizers import VQEmbeddingEMA, VectorQuantizer
from modules.decoder import DecoderPreNorm, DecoderPreNormGeneral
from modules.encoder import EncoderPreNorm
from modules.optim import ScheduledOptim

from data_loader import PathManipulationDataLoader, get_padded_sequence, PathBiManipulationDataLoader, PathManipulationFetchDataLoader

from toolz.itertoolz import partition
import argparse

from torch.utils.tensorboard import SummaryWriter


def calculate_quantization_loss(z, z_q, mask, beta):
    ''' Calcualte the quantization loss.
    :param z: Input vector, expected size (B, S, E)
    :param z_q: Quantized vector, expected size (B, S, E)
    :param mask: mask to handle uneven sequene length.
    :param beta: scalar value, scales gradients of encoder.
    '''
    total_seq = mask.sum()
    codebook_loss = (((z_q-z.detach())**2).sum(axis=-1)*mask).sum()/total_seq
    commitment_loss = (((z_q.detach()-z)**2).sum(axis=-1)*mask).sum()/total_seq
    loss = beta * commitment_loss + codebook_loss
    return loss


def calculate_reconstruction_loss(input_traj, mu, sigma, mask):
    ''' Calculates the likelihood of trajectory.
    :param input_traj:
    :param mu:
    :param sigma:
    :param mask:
    :returns torch.float:
    '''
    dist = MultivariateNormal(mu, torch.diag_embed(sigma))
    return -(dist.log_prob(input_traj)*mask).sum(dim=1).mean()


def calculate_reconstruction_loss_v2(input_traj, mu, sigma, mask, gamma):
    ''' Calculates the likelihood of trajectory and entropy.
    :param mu:
    :param sigma:
    :param mask:
    :returns torch.float: 
    '''
    # dist = MultivariateNormal(mu, torch.diag_embed(sigma))
    dist = MultivariateNormal(mu, sigma)
    neg_likelihood = -(dist.log_prob(input_traj)*mask).sum(dim=1).mean()
    neg_entropy = -(dist.entropy()*mask).sum(1).mean()
    return neg_likelihood + gamma*neg_entropy


def train_epoch(train_dataset, encoder_model, quantizer_model, decoder_model, optimizer, device):
    '''Train one epoch of the model.
    :param train_dataset:
    :param encoder_model:
    :param quantizer_model:
    :param decoder_model:
    :returns float:
    '''
    for model_i in [encoder_model, quantizer_model, decoder_model]:
        model_i.train()
    total_loss = 0
    total_reconstructional_loss = 0
    total_quantization_loss = 0
    for batch in tqdm(train_dataset, mininterval=2):
        optimizer.zero_grad()
        quantizer_model.zero_grad()
        encoder_input = batch['path'].float().to(device)
        mask = batch['mask'].to(device)
        encoder_output, = encoder_model(encoder_input)
        encoder_output_q, (_, _, indices) = quantizer_model(
            encoder_output, mask)
        quantization_loss = calculate_quantization_loss(
            encoder_output, encoder_output_q, mask, beta=0.25)
        output_dist_mu, output_dist_sigma = decoder_model(encoder_output_q)
        reconstruction_loss = calculate_reconstruction_loss_v2(
            encoder_input, output_dist_mu, output_dist_sigma, mask, gamma=0.1)
        loss = quantization_loss + reconstruction_loss
        mask_flatten = mask.view(-1)

        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()
        total_reconstructional_loss += reconstruction_loss.item()
        total_quantization_loss += quantization_loss.item()
    return total_loss, total_reconstructional_loss, total_quantization_loss


def eval_epoch(eval_dataset, encoder_model, quantizer_model, decoder_model, device):
    '''Eval one epoch of the model.
    :param eval_dataset:
    :param encoder_model:
    :param quantizer_model:
    :param decoder_model:
    :returns float:
    '''
    for model_i in [encoder_model, quantizer_model, decoder_model]:
        model_i.eval()
    total_loss = 0
    total_reconstructional_loss = 0
    total_quantization_loss = 0
    for batch in tqdm(eval_dataset, mininterval=2):
        encoder_input = batch['path'].float().to(device)
        mask = batch['mask'].to(device)
        encoder_output, = encoder_model(encoder_input)
        encoder_output_q, _ = quantizer_model(encoder_output, mask)
        quantization_loss = calculate_quantization_loss(
            encoder_output, encoder_output_q, mask, beta=0.25)
        output_dist_mu, output_dist_sigma = decoder_model(encoder_output_q)
        reconstruction_loss = calculate_reconstruction_loss_v2(
            encoder_input, output_dist_mu, output_dist_sigma, mask, gamma=0.1)
        loss = quantization_loss + reconstruction_loss
        total_loss += loss.item()
        total_reconstructional_loss += reconstruction_loss.item()
        total_quantization_loss += quantization_loss.item()
    return total_loss, total_reconstructional_loss, total_quantization_loss


def get_torch_dataloader(dataset, batch_size, num_workers):
    ''' Returns an object of type torch.data.DataLoader for the given dataset
    which will be accessed by the given number of workers.
    :param dataset: an object of type torch.data.Dataset
    :param batch_size: partition the dataset wrt the given batch size.
    :param num_workers: int, specifying number of workers.
    :return torch.data.DataLoader object.
    '''
    data_index = dataset.indexDictForest+dataset.indexDictMaze
    random.shuffle(data_index)
    batch_sampler_data = list(partition(batch_size, data_index))
    return DataLoader(dataset, num_workers=num_workers,
                      batch_sampler=batch_sampler_data, collate_fn=get_padded_sequence)


def main(args):
    ''' Train the model.
    :param args: Argument parser object
    '''
    batch_size = args.batch_size
    log_dir = args.log_dir
    num_epochs = args.num_epochs
    continue_training = args.continue_train

    model_args = dict(
        n_layers=3,
        n_heads=3,
        d_k=512,
        d_v=256,
        d_model=512,
        d_inner=1024,
        n_position=1000,
        dropout=0.1,
        c_space_dim=args.c_space_dim
    )
    with open(osp.join(log_dir, 'model_params.json'), 'w') as f:
        json.dump(model_args, f, sort_keys=True, indent=4)

    num_keys = args.num_keys
    encoder_model = EncoderPreNorm(**model_args)
    quantizer_model = VectorQuantizer(
        n_e=num_keys, e_dim=8, latent_dim=model_args['d_model'])
    decoder_model = DecoderPreNormGeneral(
        e_dim=model_args['d_model'], h_dim=model_args['d_inner'], c_space_dim=model_args['c_space_dim'])

    device = 'cpu'
    if torch.cuda.is_available():
        print("Using GPU....")
        device = torch.device('cuda')
    encoder_model.to(device)
    quantizer_model.to(device)
    decoder_model.to(device)

    optimizer = ScheduledOptim(
        t_optim.Adam(list(encoder_model.parameters())+list(quantizer_model.parameters()) +
                     list(decoder_model.parameters()), betas=(0.9, 0.98), eps=1e-9),
        lr_mul=0.2,
        d_model=512,
        n_warmup_steps=20000
    )

    # Continue learning.
    start_epoch = 0
    checkpoint_file = osp.join(log_dir, 'best_model.pkl')
    if continue_training:
        if osp.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch']
            encoder_model.load_state_dict(checkpoint['encoder_state'])
            quantizer_model.load_state_dict(checkpoint['quantizer_state'])
            decoder_model.load_state_dict(checkpoint['decoder_state'])
            optimizer._optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer.n_steps = checkpoint['n_steps']
        else:
            print(f"Cannot find file : {checkpoint_file}")

    data_folder = args.data_dir

    if args.robot=='6D':
        # Add the data loader.
        train_dataset = PathManipulationDataLoader(
            data_folder=osp.join(data_folder, 'train'),
            env_list=list(range(2000))
            )

        eval_dataset = PathManipulationDataLoader(
            data_folder=osp.join(data_folder, 'val'),
            env_list=list(range(2000, 2500))
            )
    elif args.robot=='7D':
        import fetch_utils as fu
        # Add the data loader.
        train_dataset = PathManipulationFetchDataLoader(
            data_folder=osp.join(data_folder, 'train'),
            env_list=list(range(1)),
            q_min = fu.q_min,
            q_max = fu.q_max
            )

        eval_dataset = PathManipulationFetchDataLoader(
            data_folder=osp.join(data_folder, 'val'),
            env_list=list(range(1)),
            q_min = fu.q_min,
            q_max = fu.q_max
            )        
    elif args.robot=='14D':
        # Add the data loader.
        train_dataset = PathBiManipulationDataLoader(
            data_folder=osp.join(data_folder, 'train'),
            env_list=list(range(1))
        )

        eval_dataset = PathBiManipulationDataLoader(
            data_folder=osp.join(data_folder, 'val'),
            env_list=list(range(1))
        )

    # Define the dataloader.
    training_data = DataLoader(train_dataset, num_workers=15,
                            collate_fn=get_padded_sequence, batch_size=batch_size)

    # Define the dataloader.
    evaluate_data = DataLoader(eval_dataset, num_workers=10,
                            collate_fn=get_padded_sequence, batch_size=batch_size)

    # Add the train code.
    writer = SummaryWriter(log_dir=log_dir)
    best_eval_loss = None
    for n in range(start_epoch, num_epochs):
        print(f"Epoch............: {n}")
        # Get loss of the model.
        # Index 0 - total loss
        # Index 1 - reconstructional loss
        # Index 2 - quantization loss
        train_all_losses = train_epoch(
            training_data, encoder_model, quantizer_model, decoder_model, optimizer, device)
        eval_all_losses = eval_epoch(
            evaluate_data, encoder_model, quantizer_model, decoder_model, device)

        # Periodically save trainiend model
        if (n+1) % 10 == 0:
            states = {
                'encoder_state': encoder_model.state_dict(),
                'quantizer_state': quantizer_model.state_dict(),
                'decoder_state': decoder_model.state_dict(),
                'optimizer': optimizer._optimizer.state_dict(),
                'epoch': n,
                'n_steps': optimizer.n_steps
            }
            torch.save(states, osp.join(log_dir, f'model_{n}.pkl'))
        if best_eval_loss is None:
            best_eval_loss = eval_all_losses[1]

        if eval_all_losses[1] < best_eval_loss:
            print(best_eval_loss)
            best_eval_loss = eval_all_losses[1]
            states = {
                'encoder_state': encoder_model.state_dict(),
                'quantizer_state': quantizer_model.state_dict(),
                'decoder_state': decoder_model.state_dict(),
                'optimizer': optimizer._optimizer.state_dict(),
                'epoch': n,
                'n_steps': optimizer.n_steps
            }
            torch.save(states, osp.join(log_dir, 'best_model.pkl'))

        writer.add_scalar('Loss/train', train_all_losses[0], n)
        writer.add_scalar('Reconstruct/train', train_all_losses[1], n)
        writer.add_scalar('Quantization/train', train_all_losses[2], n)
        writer.add_scalar('Loss/val', eval_all_losses[0], n)
        writer.add_scalar('Reconstruct/val', eval_all_losses[1], n)
        writer.add_scalar('Quantization/val', eval_all_losses[2], n)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help="Batch size",
                        required=True, type=int)
    parser.add_argument(
        '--num_epochs', help="Number of epochs to train the model", type=int)
    parser.add_argument(
        '--log_dir', help="Directory to save data related to training", default='')
    parser.add_argument('--continue_train',
                        help="If passed, continues model training", action='store_true')
    parser.add_argument('--c_space_dim', help="Dimension of the state space", type=int)
    parser.add_argument('--num_keys', help="Number of dictionary keys", type=int)
    parser.add_argument('--data_dir', help="Directory where training data is stored")
    parser.add_argument('--robot', help="Choose the robot model to train", choices=['2D', '6D', '14D', '7D'])

    args = parser.parse_args()

    main(args)
