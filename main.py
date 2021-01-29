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
from torchvision import datasets

LABEL_TRAIN_DATA = 'train'
UNLABEL_TRAIN_DATA = 'train+unlabeled'

LABEL_TRAIN = 'label_train'
UNLABEL_TRAIN = 'unlabel_train'
TEST = 'test'
FINE_TUNING = 'fine_tuning'
BASELINE_TRAIN = 'baseline_train'

def train(args, epochs, optimizer, criterion, scheduler, train_loader, valid_loader, model, model_name, device):
    print("Training start using unlabel dataset")
    train_loss_list, valid_loss_list = [], []
    best_loss = 987654321
    for epoch in range(epochs):
        train_total_loss, valid_total_loss = 0.0, 0.0

        model.train()
        for (xi, xj), _ in train_loader:
            zi = model(xi.to(device)) # Shape: batch * output
            zj = model(xj.to(device))

            optimizer.zero_grad()
            loss = criterion(zi, zj)
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
        if args.no_save == False and best_loss >= valid_total_loss:
            best_loss = valid_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.model_f.state_dict(),
                'projection_state_dict': model.model_g.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss
            }, "./model/" + model_name + ".pt")
            print("Model save")
    print("Training finish")
    return train_loss_list, valid_loss_list

def train_classifier(args, epochs, optimizer, criterion, train_loader, valid_loader, model, model_name, device):
    # Freeze f funciton of model
    if args.option == LABEL_TRAIN:
        for param in model.model_f.parameters():
                param.requires_grad = False

    print("Training start with label dataset")
    train_loss_list, valid_loss_list = [], []
    best_loss = 987654321
    remain_early_stopping = 10
    for epoch in range(epochs):
        if remain_early_stopping <= 0:
            break
        train_total_loss, valid_total_loss = 0.0, 0.0
        
        model.train()
        for x, label in train_loader:
            z = model(x.to(device), using_label=True)

            optimizer.zero_grad()
            loss = criterion(z, label.to(device))
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            if args.debug:
                break
        # scheduler.step()
        
        model.eval()
        with torch.no_grad():
            for val_x, label in valid_loader:
                val_z = model(val_x.to(device), using_label=True)

                loss = criterion(val_z, label.to(device))
                valid_total_loss += loss.item()
                if args.debug:
                    break

        train_loss_list.append(train_total_loss)
        valid_loss_list.append(valid_total_loss)

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) +  "|| [" + str(epoch) + "/" + str(epochs) + "], train_loss = " + str(train_total_loss) + ", valid_loss = " + str(valid_total_loss))
        
        save_name = "./model/" + model_name + "_with_label"
        if args.option == FINE_TUNING:
            save_name += '_with_fine_uning'
        save_name += '.pt'

        if args.no_save == False and (best_loss >= valid_total_loss):
            best_loss = valid_total_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': best_loss
            }, save_name)
            print("Model save")
            remain_early_stopping = 10
        else:
            remain_early_stopping -= 1

    print("Training finish")
    return train_loss_list, valid_loss_list

def test(test_loader, model, device):
    print("Test start")
    total_num, correct_num, incorrect_num = 0, 0, 0

    model.eval()
    with torch.no_grad():
        for x, label in test_loader:
            prediction = model(x.to(device), using_label=True)
            _, idxs = prediction.max(dim=1)
            total_num += x.shape[0]
            correct_num += idxs.eq(label.to(device)).sum()
            break
    
    print("Total: " + str(total_num), "Correct: " + str(correct_num), "Accuracy: " + str(float(correct_num) / float(total_num)))

def plot_loss_curve(train_loss_list, valid_loss_list, dataset, name = ''):
    # Save training log
    if not os.path.isdir(os.path.join('./result', dataset)):
        os.mkdir(os.path.join('./result', dataset))
    x_axis = np.arange(len(train_loss_list))
    plt.figure()
    plt.plot(x_axis, train_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig(os.path.join('./result', dataset, 'Train_loss' + name + '.png'))

    plt.figure()
    plt.plot(x_axis, valid_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('Valid loss')
    plt.savefig(os.path.join('./result', dataset, 'Valid_loss' + name + '.png'))


def main(args):
    ### Folder setting ###
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    if not os.path.isdir('./result'):
        os.mkdir('./result')
    if not os.path.isdir('./model'):
        os.mkdir('./model')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### Hyperparameters setting ###
    epochs = args.epochs
    batch_size = args.batch_size
    T = args.temperature
    proj_dim = args.out_dim
    
    print("Using device: " + str(device))

    dataset = DataSetWrapper(args.batch_size , args.num_worker , args.valid_size, input_shape = (96, 96, 3), dataset = args.dataset, color_distortion=args.color_distortion)
    model = make_model(512, 256, proj_dim, 10).to(device)

    '''
    Model-- ResNet18(encoder network) + MLP with one hidden layer(projection head)
    Loss -- NT-Xent Loss
    '''
    
    if args.option == UNLABEL_TRAIN:
        train_loader , valid_loader = dataset.get_data_loaders(UNLABEL_TRAIN_DATA)

        criterion = NT_XentLoss(batch_size, T, device)
        optimizer = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)

        # Load previous model & Set model name
        model_name = ''
        model_name += 'encoder_model'
        if args.prev_model != '':
            checkpoint = torch.load(args.prev_model, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            model_name = str(checkpoint['epoch']) + "-" + str(checkpoint['epoch']+epochs)

        # Train
        train_loss_list, valid_loss_list = train(args, epochs, optimizer, criterion,
                                                 scheduler, train_loader, valid_loader,
                                                 model, model_name, device)

        if not args.no_save:
            plot_loss_curve(train_loss_list, valid_loss_list, args.dataset)
    elif args.option == LABEL_TRAIN or args.option == FINE_TUNING or args.option == BASELINE_TRAIN:
        train_loader , valid_loader = dataset.get_data_loaders(LABEL_TRAIN_DATA)
        criterion = nn.CrossEntropyLoss()
        if args.option == LABEL_TRAIN:
            optimizer = torch.optim.Adam(model.linear.parameters(), 1e-4)
        elif args.option == FINE_TUNING:
            optimizer = torch.optim.Adam(list(model.linear.parameters()) + list(model.model_f.parameters()), 1e-4)
        elif args.option == BASELINE_TRAIN:
            optimizer = torch.optim.Adam(model.linear.parameters(), 1e-4)

        # Set model name
        model_name = ''
        if args.option == LABEL_TRAIN:
            model_name += 'linear_model'
        elif args.option == FINE_TUNING:
            model_name += 'fine_tuning_model'
        elif args.option == BASELINE_TRAIN:
            model_name += 'baseline_model'

        # Load previous model
        if args.option != BASELINE_TRAIN:
            assert args.prev_model != ''
            checkpoint = torch.load(args.prev_model, map_location=device)
            model.model_f.load_state_dict(checkpoint['encoder_state_dict'])
            model_name += str(checkpoint['epoch'])
        
        # Train
        train_loss_list, valid_loss_list = train_classifier(args, epochs, optimizer, criterion,
                                                            train_loader, valid_loader,
                                                            model, model_name, device)
        
        # Set plot name
        plot_name = ''
        if args.option == FINE_TUNING:
            plot_name += '_fine_tuning'
        elif args.option == BASELINE_TRAIN:
            plot_name += '_baseline'
        elif args.option == LABEL_TRAIN:
            plot_name += '_linear'

        plot_name += '_classfication' + plot_name
        if not args.no_save:
            plot_loss_curve(train_loss_list, valid_loss_list, args.dataset, name=plot_name)
    elif args.option == TEST:
        test_loader = dataset.get_test_data_loaders()

        # Load previous model
        assert args.prev_model != ''
        checkpoint = torch.load(args.prev_model, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        test(test_loader, model, device)
        
if __name__ == '__main__':
    print("TRY19")
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
        '--prev_model',
        type=str,
        default=''
    )
    parser.add_argument(
        '--option',
        type=str,
        default='unlabel_train'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='STL-10'
    )
    parser.add_argument(
        '--color_distortion',
        type=float,
        default=0.8
    )


    args = parser.parse_args()
    main(args)




