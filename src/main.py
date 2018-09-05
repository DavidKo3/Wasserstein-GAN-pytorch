# -*- coding: utf-8 -*-

from __future__ import print_function

import os, time, sys
import matplotlib.pyplot as plt
import itertools
import pickle
import imageio

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision.utils as vutils

import argparse
import dcgan as model
import data_loader as data_loader
from utils import *

import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--niter', type=int, default=25, help='number of epoch to train for')
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--manualseed', type=int, help='manual seed')
parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
opt = parser.parse_args()



try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)
# print("Random seed : ", opt.manualseed)
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)

if torch.cuda.is_available() :
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")




def show_result(num_epoch, show = False, save = False, path = 'result.png', isFix=False):
    z_ = torch.randn((5*5, 100)).view(-1, 100, 1, 1)
    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    if isFix:
        test_images = G(fixed_z_)
    else:
        test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def show_train_hist(hist, show = False, save = False, path = 'Train_hist.png'):
    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Iter')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close
# training parameters

lr = 0.00005
num_epochs = 20

isCrop = False
# data_loader
img_size = 64
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform_manualed = transforms.Compose([

    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

])

root_dir= '/home/davidk/Downloads/celeba_dataset/'
img_pwd = '/resized_celeba/celebA/'
img_random_crop_pwd = '/gaussian_noise_celeba/celebA/'
annotations_pwd = 'annotations/list_landmarks_align_celeba.txt'

transformed_celebra_dataset = data_loader.FaceLandmarksDataset(txt_file=annotations_pwd, img_dir=img_pwd,
                                           root_dir=root_dir, transform=transform_manualed)

train_loader = torch.utils.data.DataLoader(transformed_celebra_dataset, batch_size=opt.batchSize,
                                           shuffle=True, num_workers=2, drop_last=True)

transformed_celebra_random_crop_fixed = data_loader.FaceLandmarksDataset(txt_file=annotations_pwd, img_dir=img_random_crop_pwd,
                                           root_dir=root_dir, transform=transform_manualed)

random_crop_loader = torch.utils.data.DataLoader(transformed_celebra_random_crop_fixed, batch_size=opt.batchSize,
                                           shuffle=True, num_workers=2, drop_last=True)


# coco_cap = datasets.CocoCaptions(root = '/home/davidk/Downloads/cocodataset/train2014.zip',
#                         annFile = '/home/davidk/Downloads/annotations_trainval2014.zip',
#                         transform=transform_manualed)
#
# train_loader= torch.utils.data.DataLoader(coco_cap, batch_size=opt.batchSize, shuffle=True ,num_workers=2)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print("conv")
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        print("batch")
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# network
G = model.G(128)
D = model.D(128)
# mem = model.measurement()

D.apply(weights_init)
G.apply(weights_init)
# G.weight_init(G)
# D.weight_init(D)
# D.apply(model.weight_init(D))
#
G.normal_init(mean=0.0, std=0.02)
D.normal_init(mean=0.0, std=0.02)

G.cuda()
D.cuda()
# mem.cuda()


fixed_z_ = torch.randn((opt.batchSize, 100)).view(-1, 100, 1, 1)    # fixed noise
fixed_noise = None
with torch.no_grad():
    fixed_noise = Variable(fixed_z_.cuda())




# fixed_noise = torch.FloatTensor(5*5, opt.nz, 1, 1).normal_(0, 1)
# fixed_noise = Variable(fixed_noise.cuda(), volatile=True)
one = torch.FloatTensor([1])
mone = one * -1


# Binary Cross Entropy loss
BCE_loss = nn.BCELoss().cuda()



# setup optimizer
if opt.adam:
    D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
    G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
else:
    D_optimizer = optim.RMSprop(D.parameters(), lr = opt.lr)
    G_optimizer = optim.RMSprop(G.parameters(), lr = opt.lr)



# train_hist = {}
# train_hist['D_losses'] = []
# train_hist['G_losses'] = []
# train_hist['per_epoch_ptimes'] = []
# train_hist['total_ptime'] = []


print("training begin !!")

num_iter = 0
start_time = time.time()

for epoch in range(num_epochs):
    D_losses = []
    G_losses = []

    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    epoch_start_time = time.time()


    for j, y_ in enumerate(train_loader, 0):
        # print(y_.shape)
        y_ = Variable(y_.cuda())
        for i, x_ in enumerate(train_loader, 0):

            # train discriminator D
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real (log(D(x)))

            mini_batch = x_.size()[0] # len of train_loader
            y_real = torch.ones(mini_batch) # [mini_batch] or [128]

            y_fake = torch.zeros(mini_batch)  # [mini_batch]

            x_, y_real, y_fake = Variable(x_.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
            # y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())

            # print(x_.shape) # [128, 3, 64, 64]
            D_result = D(x_).squeeze()  # D(x) # [128]
            # print("D_result: ", D_result)
            D_real_loss = BCE_loss(D_result, y_real) # log(D(x))   BCE_lose([128], [128])
            D_x = D_real_loss.data.mean()


    
            # train discriminator with fake (log(1 - D(G(z)))
            z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1) # [x, 100] -> [x, 100, 1 , 1]
            z = Variable(z.cuda()) # [128 x 100 x 1 x 1]


            D_fake_decision = D(G(z)).squeeze()  # D(G(z))

            D_fake_loss = BCE_loss(D_fake_decision, y_fake)
            # D_fake_loss.backward()
            D_G_z1 = D_fake_loss.data.mean()


            # Back propagation
            D_loss = D_real_loss + D_fake_loss   # log(D(x)) + log(1- D(G(z))
            D.zero_grad()
            G.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            # train generator G
            z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1) # [mini_batch x 100] - > [mini_batch x 100 x 1 x 1]
            z = Variable(z.cuda())


            gen_image = G(z) # [128 x 3 x 64 x 64]


            D_fake_decision = D(gen_image).squeeze()
            # print("D_fake_decision: ", D_fake_decision)
            G_loss = BCE_loss(D_fake_decision, y_real)
            D_G_z2 = G_loss.data.mean()

            # Back propagation
            D.zero_grad()
            G.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # loss values
            D_losses.append(D_loss.data[0])
            G_losses.append(G_loss.data[0])

            num_iter += 1

            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch+1, num_epochs, i+1, len(train_loader), D_loss.data[0], G_loss.data[0]))


            if i % 100 == 0:
                # plot_result(G, fixed_noise, 64, i, epoch,  "./results",)
                vutils.save_image(x_.data , '%s/real_sample.png' % opt.outf)
                fake = G(fixed_noise)
                vutils.save_image(fake.data,
                                  '%s/fake_samples_epoch_%03d_%04d.png' % (opt.outf, epoch, i))


    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    # do checkpointing
    torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))



end_time = time.time()
total_ptime = end_time - start_time


print("end of training time : %d" % total_ptime)




























# # -*- coding: utf-8 -*-
#
# from __future__ import print_function
#
# import os, time, sys
# import matplotlib.pyplot as plt
# import itertools
# import pickle
# import imageio
#
# import random
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.autograd import Variable
# import torchvision.utils as vutils
#
# import argparse
# import dcgan as model
# import data_loader as data_loader
# from utils import *
#
# import scipy.ndimage as ndimage
# import matplotlib
#
# matplotlib.use('Agg')
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
# parser.add_argument('--niter', type=int, default=25, help='number of epoch to train for')
# parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
# parser.add_argument('--manualseed', type=int, help='manual seed')
# parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
# parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
# parser.add_argument('--clamp_lower', type=float, default=-0.01)
# parser.add_argument('--clamp_upper', type=float, default=0.01)
# parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
# parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
# opt = parser.parse_args()
#
# try:
#     os.makedirs(opt.outf)
# except OSError:
#     pass
#
# if opt.manualseed is None:
#     opt.manualseed = random.randint(1, 10000)
# # print("Random seed : ", opt.manualseed)
# random.seed(opt.manualseed)
# torch.manual_seed(opt.manualseed)
#
# if torch.cuda.is_available():
#     print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#
#
# def show_result(num_epoch, show=False, save=False, path='result.png', isFix=False):
#     z_ = torch.randn((5 * 5, 100)).view(-1, 100, 1, 1)
#     z_ = Variable(z_.cuda(), volatile=True)
#
#     G.eval()
#     if isFix:
#         test_images = G(fixed_z_)
#     else:
#         test_images = G(z_)
#     G.train()
#
#     size_figure_grid = 5
#     fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
#     for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
#         ax[i, j].get_xaxis().set_visible(False)
#         ax[i, j].get_yaxis().set_visible(False)
#
#     for k in range(5 * 5):
#         i = k // 5
#         j = k % 5
#         ax[i, j].cla()
#         ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
#
#     label = 'Epoch {0}'.format(num_epoch)
#     fig.text(0.5, 0.04, label, ha='center')
#     plt.savefig(path)
#
#     if show:
#         plt.show()
#     else:
#         plt.close()
#
#
# def show_train_hist(hist, show=False, save=False, path='Train_hist.png'):
#     x = range(len(hist['D_losses']))
#
#     y1 = hist['D_losses']
#     y2 = hist['G_losses']
#
#     plt.plot(x, y1, label='D_loss')
#     plt.plot(x, y2, label='G_loss')
#
#     plt.xlabel('Iter')
#     plt.ylabel('Loss')
#
#     plt.legend(loc=4)
#     plt.grid(True)
#     plt.tight_layout()
#
#     if save:
#         plt.savefig(path)
#
#     if show:
#         plt.show()
#     else:
#         plt.close
#
#
# # training parameters
#
# lr = 0.00005
# num_epochs = 20
#
# isCrop = False
# # data_loader
# img_size = 64
# if isCrop:
#     transform = transforms.Compose([
#         transforms.Scale(108),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# else:
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
# ])
#
# transform_manualed = transforms.Compose([
#
#     transforms.ToTensor(),
#     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#
# ])
#
# root_dir = '/home/davidk/Downloads/celeba_dataset/'
# img_pwd = '/resized_celeba/celebA/'
# img_random_crop_pwd = '/gaussian_noise_celeba/celebA/'
# annotations_pwd = 'annotations/list_landmarks_align_celeba.txt'
#
# transformed_celebra_dataset = data_loader.FaceLandmarksDataset(txt_file=annotations_pwd, img_dir=img_pwd,
#                                                                root_dir=root_dir, transform=transform_manualed)
#
# train_loader = torch.utils.data.DataLoader(transformed_celebra_dataset, batch_size=opt.batchSize,
#                                            shuffle=True, num_workers=2, drop_last=True)
#
# transformed_celebra_random_crop_fixed = data_loader.FaceLandmarksDataset(txt_file=annotations_pwd,
#                                                                          img_dir=img_random_crop_pwd,
#                                                                          root_dir=root_dir,
#                                                                          transform=transform_manualed)
#
# random_crop_loader = torch.utils.data.DataLoader(transformed_celebra_random_crop_fixed, batch_size=opt.batchSize,
#                                                  shuffle=True, num_workers=2, drop_last=True)
#
#
# # coco_cap = datasets.CocoCaptions(root = '/home/davidk/Downloads/cocodataset/train2014.zip',
# #                         annFile = '/home/davidk/Downloads/annotations_trainval2014.zip',
# #                         transform=transform_manualed)
# #
# # train_loader= torch.utils.data.DataLoader(coco_cap, batch_size=opt.batchSize, shuffle=True ,num_workers=2)
#
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         print("conv")
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         print("batch")
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
#
# # network
# G = model.G(128)
# D = model.D(128)
# # mem = model.measurement()
#
# D.apply(weights_init)
# G.apply(weights_init)
# # G.weight_init(G)
# # D.weight_init(D)
# # D.apply(model.weight_init(D))
# #
# G.normal_init(mean=0.0, std=0.02)
# D.normal_init(mean=0.0, std=0.02)
#
# G.cuda()
# D.cuda()
# # mem.cuda()
#
#
# fixed_z_ = torch.randn((opt.batchSize, 100)).view(-1, 100, 1, 1)  # fixed noise
# fixed_noise = None
# with torch.no_grad():
#     fixed_noise = Variable(fixed_z_.cuda())
#
# # fixed_noise = torch.FloatTensor(5*5, opt.nz, 1, 1).normal_(0, 1)
# # fixed_noise = Variable(fixed_noise.cuda(), volatile=True)
# one = torch.FloatTensor([1])
# mone = one * -1
#
# # Binary Cross Entropy loss
# BCE_loss = nn.BCELoss().cuda()
#
# # setup optimizer
# if opt.adam:
#     D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(0.5, 0.999))
#     G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(0.5, 0.999))
# else:
#     D_optimizer = optim.RMSprop(D.parameters(), lr=opt.lr)
#     G_optimizer = optim.RMSprop(G.parameters(), lr=opt.lr)
#
# # train_hist = {}
# # train_hist['D_losses'] = []
# # train_hist['G_losses'] = []
# # train_hist['per_epoch_ptimes'] = []
# # train_hist['total_ptime'] = []
#
#
# print("training begin !!")
#
# num_iter = 0
# start_time = time.time()
#
# for epoch in range(num_epochs):
#     D_losses = []
#     G_losses = []
#
#     # learning rate decay
#     if (epoch + 1) == 11:
#         G_optimizer.param_groups[0]['lr'] /= 10
#         D_optimizer.param_groups[0]['lr'] /= 10
#         print("learning rate change!")
#
#     if (epoch + 1) == 16:
#         G_optimizer.param_groups[0]['lr'] /= 10
#         D_optimizer.param_groups[0]['lr'] /= 10
#         print("learning rate change!")
#
#     epoch_start_time = time.time()
#
#     for j, y_ in enumerate(train_loader, 0):
#         # print(y_.shape)
#         y_ = Variable(y_.cuda())
#         for i, x_ in enumerate(train_loader, 0):
#
#             # train discriminator D
#             ############################
#             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#             ###########################
#             # train with real (log(D(x)))
#
#             mini_batch = x_.size()[0]  # len of train_loader
#             y_real = torch.ones(mini_batch)  # [mini_batch] or [128]
#
#             y_fake = torch.zeros(mini_batch)  # [mini_batch]
#
#             x_, y_real, y_fake = Variable(x_.cuda()), Variable(y_real.cuda()), Variable(y_fake.cuda())
#             # y_real, y_fake = Variable(y_real.cuda()), Variable(y_fake.cuda())
#
#             # print(x_.shape) # [128, 3, 64, 64]
#             D_result = D(x_).squeeze()  # D(x) # [128]
#             # print("D_result: ", D_result)
#             D_real_loss = BCE_loss(D_result, y_real)  # log(D(x))   BCE_lose([128], [128])
#             D_x = D_real_loss.data.mean()
#
#             # train discriminator with fake (log(1 - D(G(z)))
#             z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)  # [x, 100] -> [x, 100, 1 , 1]
#             z = Variable(z.cuda())  # [128 x 100 x 1 x 1]
#
#             D_fake_decision = D(G(z)).squeeze()  # D(G(z))
#
#             D_fake_loss = BCE_loss(D_fake_decision, y_fake)
#             # D_fake_loss.backward()
#             D_G_z1 = D_fake_loss.data.mean()
#
#             # Back propagation
#             D_loss = D_real_loss + D_fake_loss  # log(D(x)) + log(1- D(G(z))
#             D.zero_grad()
#             G.zero_grad()
#             D_loss.backward()
#             D_optimizer.step()
#
#             ############################
#             # (2) Update G network: maximize log(D(G(z)))
#             ###########################
#             # train generator G
#             z = torch.randn((mini_batch, 100)).view(-1, 100, 1, 1)  # [mini_batch x 100] - > [mini_batch x 100 x 1 x 1]
#             z = Variable(z.cuda())
#
#             gen_image = G(z)  # [128 x 3 x 64 x 64]
#
#             D_fake_decision = D(gen_image).squeeze()
#             # print("D_fake_decision: ", D_fake_decision)
#             G_loss = BCE_loss(D_fake_decision, y_real)
#             D_G_z2 = G_loss.data.mean()
#
#             # Back propagation
#             D.zero_grad()
#             G.zero_grad()
#             G_loss.backward()
#             G_optimizer.step()
#
#             # loss values
#             D_losses.append(D_loss.data[0])
#             G_losses.append(G_loss.data[0])
#
#             num_iter += 1
#
#             print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
#                   % (epoch + 1, num_epochs, i + 1, len(train_loader), D_loss.data[0], G_loss.data[0]))
#
#             if i % 100 == 0:
#                 # plot_result(G, fixed_noise, 64, i, epoch,  "./results",)
#                 vutils.save_image(x_.data, '%s/real_sample.png' % opt.outf)
#                 fake = G(fixed_noise)
#                 vutils.save_image(fake.data,
#                                   '%s/fake_samples_epoch_%03d_%04d.png' % (opt.outf, epoch, i))
#
#     epoch_end_time = time.time()
#     per_epoch_ptime = epoch_end_time - epoch_start_time
#     # do checkpointing
#     torch.save(G.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
#     torch.save(D.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
#
# end_time = time.time()
# total_ptime = end_time - start_time
#
# print("end of training time : %d" % total_ptime)

