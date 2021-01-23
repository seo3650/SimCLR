import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from data_loader import DataSetWrapper
from simCLR import make_model, NT_XentLoss
import time
import matplotlib.pyplot as plt


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Hyperparameters setting ###
    epochs = args.epochs
    batch_size = args.batch_size
    T = args.temperature
    proj_dim = args.out_dim
    
    print("Using device: " + str(device))

    ### DataLoader ###
    dataset = DataSetWrapper(args.batch_size , args.num_worker , args.valid_size, input_shape = (96, 96, 3))
    train_loader , valid_loader = dataset.get_data_loaders()


    ### You may use below optimizer & scheduler ###
    model = make_model(1000, 256, proj_dim).to(device)
    criterion = NT_XentLoss(batch_size, T, device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

    # Load previous model
    model_name = '0-' + str(epochs)
    if (args.prev_model_dir != ''):
        checkpoint = torch.load(args.prev_model_dir, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        model_name = str(checkpoint['epoch']) + "-" + str(checkpoint['epoch']+epochs)
    '''
    Model-- ResNet18(encoder network) + MLP with one hidden layer(projection head)
    Loss -- NT-Xent Loss
    '''
    

    print("Training start")
    train_loss_list, valid_loss_list = [], []
    for epoch in range(epochs):
        train_total_loss, valid_total_loss = 0.0, 0.0
        best_loss = 987654321

        model.train()
        for (xi, xj), _ in train_loader:
            zi = model(xi.to(device)) # Shape: batch * output
            zj = model(xj.to(device))

            loss = criterion(zi, zj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            if args.debug:
                break
        
        scheduler.step()
        # You have to save the model using early stopping
        model.eval()
        with torch.no_grad():
            for (val_xi, val_xj), _ in valid_loader:
                val_zi = model(val_xi.to(device))
                val_zj = model(val_xj.to(device))

                loss = criterion(val_zi, val_zj)
                valid_total_loss += loss.item()
                if args.debug:
                    break
        
        train_loss_list.append(train_total_loss)
        valid_loss_list.append(valid_total_loss)

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +  "|| [" + str(epoch) + "/" + str(epochs) + "], train_loss = " + str(train_total_loss) + ", valid_loss = " + str(valid_total_loss))
        if args.no_save == False and best_loss > valid_total_loss:
            best_loss = valid_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss
            }, "./best_model" + model_name + ".pt")
            print("Model save")
        

    print("Training finish")
    
    # Save training log
    x_axis = np.arange(epochs)
    plt.figure()
    plt.plot(x_axis, train_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('./result/Trinin_loss.png')

    plt.figure()
    plt.plot(x_axis, valid_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Valid loss')
    plt.savefig('./result/Valid_loss.png')

if __name__ == '__main__':
    print("TRY7")
    parser = argparse.ArgumentParser(description = "SimCLR implementation")

    parser.add_argument(
        '--epochs',
        type=int,
        default=40)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--temperature',
        type=float,
        default=0.5)
    parser.add_argument(
        '--out_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--num_worker',
        type=int,
        default=8)

    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05)
    parser.add_argument(
        '--debug',
        action='store_true'
    )
    parser.add_argument(
        '--no_save',
        action='store_true'
    )
    parser.add_argument(
        '--prev_model_dir',
        type=str,
        default=''
    )

    args = parser.parse_args()
    main(args)




