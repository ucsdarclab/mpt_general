''' Code to train the model.
'''
import torch
import torch.nn as nn
from tqdm import tqdm
import os.path as osp

from torch.utils.data import DataLoader

import argparse 
from torch.utils.tensorboard import SummaryWriter

from mpnet_models import MLP, Encoder
from data_loader import MPNetDataLoader, MPNet14DDataLoader, get_mpnet_padded_seq


def train_epoch(train_dataset, encoder, mlp, criterion, optimizer, device):
    ''' Train a single epoch.

    '''
    for model in [encoder, mlp]:
        model.train()
    
    avg_loss = 0
    for batch in tqdm(train_dataset, mininterval=2):
        # forward pass through encoder
        h = encoder(batch['env'].to(device))

        for i in range(h.shape[0]):
            seq_len = batch['input_pos'].shape[1]
            # concatenate encoder output with dataset input
            inp = torch.cat((batch['input_pos'][i].to(device), h[i].expand(seq_len, -1)), dim=1)

            # forward pass through mlp
            bo = mlp(inp)
            mask = batch['mask'][i][:, None].to(device)
            # compute overall loss and backprop all the way
            loss = criterion(bo*mask, batch['target_pos'][i].to(device)*mask)
            avg_loss = avg_loss+loss.data
        loss.backward()
        optimizer.step()

    return avg_loss

def val_epoch(val_dataset, encoder, mlp, criterion, device):
    ''' Train a single epoch.

    '''
    for model in [encoder, mlp]:
        model.eval()
    
    avg_loss = 0
    for batch in tqdm(val_dataset, mininterval=2):
        # forward pass through encoder
        h = encoder(batch['env'].to(device))

        for i in range(h.shape[0]):
            seq_len = batch['input_pos'].shape[1]
            # concatenate encoder output with dataset input
            inp = torch.cat((batch['input_pos'][i].to(device), h[i].expand(seq_len, -1)), dim=1)

            # forward pass through mlp
            bo = mlp(inp)
            mask = batch['mask'][i][:, None].to(device)
            # compute overall loss and backprop all the way
            loss = criterion(bo*mask, batch['target_pos'][i].to(device)*mask)
            avg_loss = avg_loss+loss.data

    return avg_loss

device = 'cpu'
if torch.cuda.is_available():
    print("Using GPU....")
    device = torch.device('cuda')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, default='./data/train/paths/')

    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--enc_input_size', type=int, default=16053)
    parser.add_argument('--enc_output_size', type=int, default=60)
    parser.add_argument('--mlp_output_size', type=int, default=7)
    parser.add_argument('--robot', help="Choose the robot model to train", choices=['6D', '14D'])
    parser.add_argument('--cont', help="Continue training the model", action='store_true')

    args = parser.parse_args()

    # Define the models.
    mlp = MLP(args.enc_output_size+args.mlp_output_size*2, args.mlp_output_size)
    encoder = Encoder(args.enc_input_size, args.enc_output_size)

    for models in [mlp, encoder]:
        models.to(device)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    params = list(encoder.parameters())+list(mlp.parameters())
    optimizer = torch.optim.Adagrad(params, lr=args.learning_rate)

    if args.robot=='6D':
        # Create dataset objects.
        train_dataset = MPNetDataLoader('/root/data/pandav3/train', list(range(2000)), args.enc_input_size//3)
        val_dataset = MPNetDataLoader('/root/data/pandav3/val', list(range(2000, 2500)), args.enc_input_size//3)
    elif args.robot=='14D':
        train_dataset = MPNet14DDataLoader('/root/data/bi_panda/train', list(range(1, 2000)), args.enc_input_size//3)
        val_dataset = MPNet14DDataLoader('/root/data/bi_panda/val', list(range(2001, 2500)), args.enc_input_size//3)

    # Writing data loader
    train_dataloader = DataLoader(train_dataset, collate_fn=get_mpnet_padded_seq, num_workers=10, shuffle=True, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, num_workers=5, collate_fn=get_mpnet_padded_seq, shuffle=True, batch_size=args.batch_size)

    writer = SummaryWriter(log_dir=args.log_dir)
    start_epoch = 0
    if args.cont:
        checkpoint = torch.load(osp.join(args.log_dir, 'model_99.pkl'))
        mlp.load_state_dict(checkpoint['mlp_state'])
        encoder.load_state_dict(checkpoint['encoder_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    for epoch_i in range(start_epoch+1, args.num_epochs):

        # Train the model for a single epoch.
        print(f"epoch {epoch_i}")
        train_avg_loss = train_epoch(train_dataloader, encoder, mlp, criterion, optimizer, device)
        val_avg_loss = val_epoch(val_dataloader, encoder, mlp, criterion, device)

        # Periodically save trainiend model
        if (epoch_i+1) % 10 == 0:
            states = {
                'encoder_state': encoder.state_dict(),
                'mlp_state': mlp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_i
            }
            torch.save(states, osp.join(args.log_dir, f'model_{epoch_i}.pkl'))
        
        writer.add_scalar('Loss/train', train_avg_loss, epoch_i)
        writer.add_scalar('Loss/val', val_avg_loss, epoch_i)
