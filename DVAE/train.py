import os
import sys
import math
import pickle
import pdb
import argparse
import random
from tqdm import tqdm
from shutil import copy
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import scipy.io
from scipy.linalg import qr
import igraph
from random import shuffle
import matplotlib
from sklearn import manifold

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from util import *
from models_ff3_fb2 import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings
parser.add_argument('--data-name', default='threeStageOpamp', help='graph dataset name')
parser.add_argument('--save-appendix', default='',
                    help='what to append to data-name as save-name for results')
parser.add_argument('--only-test', action='store_true', default=False,
                    help='if True, perform some experiments without training the model')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait each time to save model states')
parser.add_argument('--sample-number', type=int, default=10, metavar='N',
                    help='how many samples to generate each time')
parser.add_argument('--gpu', type=int, default=3, help='which gpu to use')
# training settings
parser.add_argument('--model', default='DVAE', help='model to use')
parser.add_argument('--n_topo', type=int, default=100, metavar='N',
                    help='number of topologies in the training set')
parser.add_argument('--trainset_size', type=int, default=10000, help='control the size of training set')
parser.add_argument('--hs', type=int, default=501, metavar='N',
                    help='hidden size of GRUs')
parser.add_argument('--nz', type=int, default=10, metavar='N',
                    help='number of dimensions of latent vectors z')
parser.add_argument('--load_model_name', default='500', help='model name to loaded')
# optimization settings
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size during training')
parser.add_argument('--infer-batch-size', type=int, default=128, metavar='N',
                    help='batch size during inference')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args()
torch.manual_seed(args.seed)
gpu = 'cuda:' + str(args.gpu)
device = torch.device(gpu if torch.cuda.is_available() else 'cpu')
print(device)
np.random.seed(args.seed)
random.seed(args.seed)
print(args)

'''Prepare data'''
args.file_dir = os.getcwd()
topo_num = args.n_topo

args.res_dir = os.path.join(args.file_dir,
                            'results/results_t{}_e{}/{}{}'.format(topo_num, args.epochs, args.data_name,
                                                                  args.save_appendix))

if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)
pkl_name = os.path.join(args.res_dir, args.data_name + '.pkl')
train_data, test_data, graph_args = load_circuit_graphs(topo_num)
train_data = train_data[:args.trainset_size]

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

'''Prepare the model'''
# model
model = eval(args.model)(
    max_n=13,
    # fs=graph_args.node_feature,
    nvt=12,
    START_TYPE=9,
    END_TYPE=10,
    GND_TYPE=11,
    NEG_TYPE=7,
    POS_TYPE=8,
    hs=args.hs,
    nz=args.nz
)
# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=20, verbose=True)

model.to(device)

'''Define some train/test functions'''


def train(epoch):
    model.train()
    train_loss = 0
    recon_loss = 0
    kld_loss = 0
    pbar = tqdm(train_data)
    g_batch = []

    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            optimizer.zero_grad()
            g_batch = model._collate_fn(g_batch)
            '''
            mu, logvar = model.encode(g_batch)
            loss, recon, kld = model.loss(mu, logvar, g_batch)
            '''
            loss, recon, vll, kld = model(g_batch)

            pbar.set_description(
                'Epoch: %d, loss: %0.4f, recon: %0.4f, type: %0.4f, kld: %0.4f' % (
                    epoch, loss.item() / len(g_batch), recon.item() / len(g_batch),
                    vll.item() / len(g_batch), kld.item() / len(g_batch)))
            loss.backward()
            train_loss += loss.item()
            recon_loss += recon.item()
            kld_loss += kld.item()
            optimizer.step()
            g_batch = []
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_data)))

    return train_loss, recon_loss, kld_loss


def test():
    # test recon accuracy
    test_model.eval()
    decode_times = 5
    Nll = 0
    softmax = 0
    n_perfect = 0
    print('Testing begins...')

    print('Performance on the train data: ')
    pbar1 = tqdm(train_data)
    g_batch = []
    for i, g in enumerate(pbar1):
        g_batch.append(g)
        if len(g_batch) == args.infer_batch_size or i == len(train_data) - 1:
            g = test_model._collate_fn(g_batch)
            mu, logvar = test_model.encode(g)
            _, nll, softmax_error, _ = test_model.loss(mu, logvar, g)
            pbar1.set_description('recon loss: {:.4f}'.format(nll.item() / len(g_batch)))
            Nll += nll.item()
            softmax += softmax_error.item()

            # construct igraph g from tensor g to check recon quality
            z = test_model.reparameterize(mu, logvar)
            for _ in range(decode_times):
                g_recon = test_model.decode(z, stochastic=False)
                n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))
            g_batch = []
    Nll /= len(train_data)
    softmax /= len(train_data)

    acc = n_perfect / (len(train_data) * decode_times)
    print(
        'Trainset average recon loss: {0}, softmax loss: {1}, recon accuracy: {2:.4f}'.format(
            Nll, softmax, acc))

    print('Performence on the test data: ')
    pbar = tqdm(test_data)
    g_batch = []
    Nll = 0
    softmax = 0
    n_perfect = 0
    for i, g in enumerate(pbar):
        g_batch.append(g)
        if len(g_batch) == args.infer_batch_size or i == len(test_data) - 1:
            g = test_model._collate_fn(g_batch)
            mu, logvar = test_model.encode(g)
            _, nll, softmax_error, _ = test_model.loss(mu, logvar, g)
            pbar.set_description('recon loss: {:.4f}'.format(nll.item() / len(g_batch)))
            Nll += nll.item()
            softmax += softmax_error.item()
            z = test_model.reparameterize(mu, logvar)
            for _ in range(decode_times):
                g_recon = test_model.decode(z, stochastic=False)
                n_perfect += sum(is_same_DAG(g0, g1) for g0, g1 in zip(g, g_recon))

            g_batch = []
            y_batch = []
    Nll /= len(test_data)
    softmax /= len(test_data)
    acc = n_perfect / (len(test_data) * decode_times)
    print(
        'Testset average recon loss: {0}, softmax loss: {1}, recon accuracy: {2:.4f}'.format(
            Nll, softmax, acc))


def extract_latent(data):
    model.eval()
    Z = []
    # Y = []
    g_batch = []
    for i, g in enumerate(tqdm(data)):
        # copy igraph
        # otherwise original igraphs will save the H states and consume more GPU memory
        g_ = g.copy()
        g_batch.append(g_)
        if len(g_batch) == args.infer_batch_size or i == len(data) - 1:
            g_batch = model._collate_fn(g_batch)
            mu, _ = model.encode(g_batch)
            mu = mu.cpu().detach().numpy()
            Z.append(mu)
            g_batch = []
    return np.concatenate(Z, 0)


def save_latent_representations(epoch):
    Z_train = extract_latent(train_data)
    Z_test = extract_latent(test_data)
    latent_pkl_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.pkl'.format(epoch))
    latent_mat_name = os.path.join(args.res_dir, args.data_name +
                                   '_latent_epoch{}.mat'.format(epoch))
    with open(latent_pkl_name, 'wb') as f:
        pickle.dump((Z_train, Z_test), f)
    print('Saved latent representations to ' + latent_pkl_name)
    scipy.io.savemat(latent_mat_name,
                     mdict={
                         'Z_train': Z_train,
                         'Z_test': Z_test
                     }
                     )


'''Training begins here'''
if args.only_test:
    '''Only testing'''
    load_model_name = 'model_checkpoint' + args.load_model_name + '.pth'
    test_model = torch.load(os.path.join(args.res_dir, load_model_name))
    print('model: {} has been loaded'.format(os.path.join(args.res_dir, load_model_name)))
    test()
else:
    min_loss = math.inf
    min_loss_epoch = None
    loss_name = os.path.join(args.res_dir, 'train_loss.txt')
    loss_plot_name = os.path.join(args.res_dir, 'train_loss_plot.pdf')
    test_results_name = os.path.join(args.res_dir, 'test_results.txt')

    if os.path.exists(loss_name):
        os.remove(loss_name)
    for epoch in range(1, args.epochs + 1):
        train_loss, recon_loss, kld_loss = train(epoch)
        with open(loss_name, 'a') as loss_file:
            loss_file.write("{:.2f} {:.2f} {:.2f}\n".format(
                train_loss / len(train_data),
                recon_loss / len(train_data),
                kld_loss / len(train_data),
            ))
        scheduler.step(train_loss)
        if epoch % args.save_interval == 0:
            print("save current model...")
            model_name = os.path.join(args.res_dir, 'model_checkpoint{}.pth'.format(epoch))
            optimizer_name = os.path.join(args.res_dir, 'optimizer_checkpoint{}.pth'.format(epoch))
            scheduler_name = os.path.join(args.res_dir, 'scheduler_checkpoint{}.pth'.format(epoch))
            torch.save(model, model_name)
            torch.save(optimizer.state_dict(), optimizer_name)
            torch.save(scheduler.state_dict(), scheduler_name)
            print("extract latent representations...")
            save_latent_representations(epoch)
            print("plot train loss...")
            losses = np.loadtxt(loss_name)
            if losses.ndim == 1:
                continue
            fig = plt.figure()
            num_points = losses.shape[0]
            plt.plot(range(1, num_points + 1), losses[:, 0], label='Total')
            plt.plot(range(1, num_points + 1), losses[:, 1], label='Recon')
            plt.plot(range(1, num_points + 1), losses[:, 2], label='KLD')
            plt.xlabel('Epoch')
            plt.ylabel('Train loss')
            plt.legend()
            plt.savefig(loss_plot_name)
