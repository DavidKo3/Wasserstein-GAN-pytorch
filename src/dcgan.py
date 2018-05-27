import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import scipy
import scipy.ndimage as ndimage
import imageutils as imgutil
from torchvision import  transforms

# def weight_init(self, m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
class G(nn.Module):
    def __init__(self, d=128):
        super(G, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)



    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        # print("G class for x ", x.shape)
        return x # [128, 3, 64, 64]

    def weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def normal_init(m, mean=0.0, std=0.02):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

        if isinstance(m , nn.BatchNorm2d):
            m.weight.data.normal_(1.0, std)
            m.bias.data.zero_()





class D(nn.Module):
    def __init__(self, d=128):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # def weight_init(self, mean, std):
    #     for m in self._modules:
    #         normal_init(self._modules[m], mean, std)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))
        # print("D class for x ", x.shape)
        return x # [128 x 1 x  1 x 1]

    # def weight_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         print("weight_init")
    #         m.weight.data.normal_(0.0, 0.02)
    #     elif classname.find('BatchNorm') != -1:
    #         m.weight.data.normal_(1.0, 0.02)
    #         m.bias.data.fill_(0)

    def normal_init(m, mean=0.0, std=0.02):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):

            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()

        if isinstance(m , nn.BatchNorm2d):
            m.weight.data.normal_(1.0, std)
            m.bias.data.zero_()

class measurement(nn.Module):
    """f :convolve with adding noise"""
    def __init__(self, d=64):
        super(measurement, self).__init__()
        # self.linear = nn.Linear(3*64*64, 3*64*64) # [12288 x 12288]
        self.conv = nn.Conv2d(3,3, 3, padding=1)

    def forward(self, input):
        g = self.makeGaussian(3, 1)
        batchSize = 128

        kernel = torch.FloatTensor(g)

        kernel = torch.stack(
            [kernel for i in range(3)])  # this stacks the kernel into 3 identical 'channels' for rgb images
                                         # [3 x 3 x 3]
        batched_gaussian = Variable(torch.stack([kernel for i in range(batchSize)])).cuda()  # stack kernel into batches # [128, 3, 3, 3]

        self.conv.weight.data.copy_(kernel)

        x = self.conv(input)

        # for batch in range(batchSize):
        #     for ch in range(3):
        #         t= F.ma
        #         x[batch, ch, :, :] = x[batch, ch, :, :]/ torch.max(x[batch, ch, :, :])
        # "need to nomalize for x[:,0], x[:,1], x[:,2]"
        torch.nn.functional.normalize(x, p=2, dim=2)


        # torch.nn.functional.normalize(x, p=2, dim=0)
        # x = F.conv2d(input, batched_gaussian)  # nnf is torch.nn.functional
        # print(self.conv.weight.data)

        return x # [128 x 3 x  64 x 64]

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def makeGaussian(self, size, fwhm=1, center=None):
        x = np.arange(0, size, 1, dtype=np.float32)
        y = x[:, np.newaxis]

        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]

        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)





