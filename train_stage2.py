''' Script to train Stage 2
'''

import torch
import torch.nn.functional as F
import torch.optim as t_optim

import argparse
import random
import json

from os import path as osp
from torch.utils.data import DataLoader
from toolz import partition
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from modules.autoregressive import AutoRegressiveModel, EnvContextCrossAttModel
from modules.quantizers import VectorQuantizer
from modules.optim import ScheduledOptim
from data_loader import get_quant_padded_sequence, QuantPathMixedDataLoader
from data_loader import QuantManipulationDataLoader, get_quant_manipulation_sequence

def calculate_loss(context_output, ar_model, batch_data, batch_size, device):
    ''' Calculates loss for each trajectory by training the auto-regressive model to maximize
    the likelihood for each trajectory.

    '''
    loss = 0
    total_num_trajectories  = batch_data['target_seq_id'].shape[0]
    for i in range(total_num_trajectories):
        offset = max(int((batch_data['length'][i])/batch_size), 1)
        total_length = min(batch_size*offset, int(batch_data['length'][i])-1)
        label = batch_data['target_seq_id'][i, :total_length:offset]
        batch_size_i = label.shape[0]
        
        ar_model_input_i = torch.cat([context_output[i, :, :], batch_data['input_seq'][i, :total_length, :].to(device)])
        mask  = torch.tril(torch.ones(total_length, total_length+2), diagonal=2)
        mask  = mask[::offset, :].to(device)

        target_value_index = (mask.sum(dim=1)-1).to(dtype=torch.int64)
        tmp_output = ar_model(ar_model_input_i.repeat((batch_size_i, 1, 1)), mask)
        tmp_prob_output = -1*F.log_softmax(tmp_output, dim=-1)
        
        loss +=tmp_prob_output[torch.arange(batch_size_i, device=device), target_value_index, label].sum()
    return loss/total_num_trajectories


def train_epoch(context_env_encoder, ar_model, train_dataset, batch_size, optimizer, device):
    ''' Train the model for an epoch
    :param context_env_encoder: model for encoding environment w/ start & goal pairs.
    :param ar_model: model for autoregressive models.
    :param batch_size: number of chunks each trajectory should be split to.
    :param optimizer: the schedule optimizer object.
    :param device: The device on which to train the model
    '''
    for model_i in [context_env_encoder, ar_model]:
        model_i.train()
    
    total_loss = 0
    for batch_data in tqdm(train_dataset, mininterval=2):
        optimizer.zero_grad()
        context_output = context_env_encoder(batch_data['map'].to(device), batch_data['start_n_goal'].to(device))
        loss = calculate_loss(context_output, ar_model, batch_data, batch_size, device)
        loss.backward()
        optimizer.step_and_update_lr()
        total_loss += loss.item()

    return total_loss


# Evaluate the model  once.
def eval_epoch(context_env_encoder, ar_model, eval_dataset, batch_size, device):
    ''' Evaluate the model for an epoch
    :param context_env_encoder: model for encoding environment w/ start & goal pairs.
    :param ar_model: model for autoregressive models.
    :param batch_size: number of chunks each trajectory should be split to.
    :param device: The device on which to train the model
    '''
    for model_i in [context_env_encoder, ar_model]:
        model_i.eval()
    
    total_loss = 0
    for batch_data in tqdm(eval_dataset, mininterval=2):
        with torch.no_grad():
            context_output = context_env_encoder(batch_data['map'].to(device), batch_data['start_n_goal'].to(device))
            loss = calculate_loss(context_output, ar_model, batch_data, batch_size, device)
        total_loss += loss.item()
    return total_loss


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
        batch_sampler=batch_sampler_data, collate_fn=get_quant_padded_sequence)

# define main training routine
def main(args):
    ''' Main training routine for statge 2
    '''
    dictionary_model_folder = args.dict_model_folder
    train_model_folder = args.log_dir
    batch_size = args.batch_size

    # Load the qunatizer model
    d_model=512
    num_keys = 1024
    quantizer_model = VectorQuantizer(n_e=num_keys, e_dim=8, latent_dim=d_model)
    checkpoint = torch.load(osp.join(dictionary_model_folder, 'best_model.pkl'))
    quantizer_model.load_state_dict(checkpoint['quantizer_state'])
    
    # Define Cross attention model
    if args.robot == '2D':
        env_params = {
        'd_model': d_model,
        'dropout': 0.1,
        'n_position': 40*40
        }

        context_params = dict(
        d_context=2,
        n_layers=3,
        n_heads=3, 
        d_k=512,
        d_v=256, 
        d_model=d_model, 
        d_inner=1024,
        dropout=0.1
        )
    if args.robot == '6D':
        env_params = dict(d_model=d_model)

        context_params = dict(
        d_context=6,
        n_layers=3,
        n_heads=3, 
        d_k=512,
        d_v=256, 
        d_model=512, 
        d_inner=1024,
        dropout=0.1
        )

    context_env_encoder = EnvContextCrossAttModel(env_params, context_params, robot=args.robot)
    # Save the parameters used to define AR model.
    with open(osp.join(train_model_folder, 'cross_attn.json'), 'w') as f:
        json.dump(context_params, f, sort_keys=True, indent=4)

    ar_params = dict(
    d_k = 512,
    d_v = 256,
    d_model = d_model,
    d_inner = 1024,
    dropout = 0.1,
    n_heads = 3,
    n_layers = 3,
    num_keys=num_keys+2 # +2 for encoding start and goal keys
    )
    ar_model = AutoRegressiveModel(**ar_params)

    # Save the parameters used to define AR model.
    with open(osp.join(train_model_folder, 'ar_params.json'), 'w') as f:
        json.dump(ar_params, f, sort_keys=True, indent=4)

    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    context_env_encoder.to(device)
    ar_model.to(device)

    optimizer = ScheduledOptim(
        t_optim.Adam(list(context_env_encoder.parameters()) + list(ar_model.parameters()), betas=(0.9, 0.98), eps=1e-9),
        lr_mul=0.2,
        d_model=512,
        n_warmup_steps=2400
    )

    if args.robot=='2D':
        # Define the train dataloader
        train_dataset = QuantPathMixedDataLoader(
            quantizer_model, 
            list(range(750))+list(range(1000, 1750)), 
            '/root/data2d/maze4/train',
            osp.join(dictionary_model_folder, 'quant_key/maze4/train'),
            list(range(1500)),
            '/root/data2d/forest/train',
            osp.join(dictionary_model_folder, 'quant_key/forest/train')
        )

        train_data_loader = get_torch_dataloader(train_dataset, batch_size, num_workers=20)

        # Define the eval dataloader
        val_dataset = QuantPathMixedDataLoader(
            quantizer_model, 
            list(range(500)), 
            '/root/data2d/maze4/val',
            osp.join(dictionary_model_folder, 'quant_key/maze4/val'),
            list(range(500)),
            '/root/data2d/forest/val',
            osp.join(dictionary_model_folder, 'quant_key/forest/val')
        )

        val_data_loader = get_torch_dataloader(val_dataset, batch_size, num_workers=10)

    if args.robot == '6D':
        train_dataset = QuantManipulationDataLoader(
            quantizer_model, 
            list(range(1000)),
            '/root/data/pandav3/train/',
            '/root/data/general_mpt/model1/quant_key/pandav3/train/'
        )
        train_data_loader = DataLoader(train_dataset, num_workers=15, batch_size=batch_size, collate_fn=get_quant_manipulation_sequence)

        val_dataset = QuantManipulationDataLoader(
            quantizer_model,
            list(range(2000, 2500)),
            '/root/data/pandav3/val',
            '/root/data/general_mpt/model1/quant_key/pandav3/val'
        )
        val_data_loader = DataLoader(val_dataset, num_workers=10, batch_size=batch_size, collate_fn=get_quant_manipulation_sequence)
    
    writer = SummaryWriter(log_dir=train_model_folder)
    best_eval_loss = 1e10
    start_epoch = 0
    if args.cont:
        checkpoint = torch.load(osp.join(train_model_folder, 'best_model.pkl'))
        ar_model.load_state_dict(checkpoint['ar_model_state'])
        context_env_encoder.load_state_dict(checkpoint['context_state'])
        optimizer._optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        optimizer.n_steps = checkpoint['n_steps']

    for n in range(start_epoch, args.num_epochs):
        # One valing pass of the model.
        print(f"Epoch: .......{n}")
        train_loss = train_epoch(context_env_encoder, ar_model, train_data_loader, 40, optimizer, device)
        eval_loss = eval_epoch(context_env_encoder, ar_model, val_data_loader, 40, device)
    
        # Periodically save trainiend model
        if (n+1) % 10 == 0:
            states = {
                'context_state': context_env_encoder.state_dict(),
                'ar_model_state': ar_model.state_dict(),
                'optimizer': optimizer._optimizer.state_dict(),
                'epoch': n,
                'n_steps': optimizer.n_steps
            }
            torch.save(states, osp.join(train_model_folder, f'model_{n}.pkl'))

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            print(best_eval_loss)
            states = {
                'context_state': context_env_encoder.state_dict(),
                'ar_model_state': ar_model.state_dict(),
                'optimizer': optimizer._optimizer.state_dict(),
                'epoch': n,
                'n_steps': optimizer.n_steps
            }
            torch.save(states, osp.join(train_model_folder, 'best_model.pkl'))
        
        writer.add_scalar('Loss/train', train_loss, n)
        writer.add_scalar('Loss/test', eval_loss, n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict_model_folder', help="Folder to find the dictionary model parameters")
    parser.add_argument('--num_epochs', help="Number of epochs to train the model", type=int)
    parser.add_argument('--log_dir', help="Directory to save data related to training")
    parser.add_argument('--batch_size', help="Number of trajectories to load in each batch", type=int)
    parser.add_argument('--cont', help="Continue training the model", action='store_true')
    parser.add_argument('--robot', help="Choose the robot model to train", choices=['2D', '6D'])

    args = parser.parse_args()

    main(args)