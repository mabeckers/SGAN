import argparse
import math

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, image_size: int = 28, out_channels: int = 1):
        super(Generator, self).__init__()
        self.init_size = int(np.ceil(image_size / 4)) # initial starting size of the generater image
        self.out_channels = out_channels
        self.l1 = nn.Linear(latent_dim, int(128 * self.init_size  ** 2))

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)  #(trainsamples, channels, image size lenght, image size height)
        x = nn.BatchNorm2d(128)(x)
        x = nn.Upsample(scale_factor=2)(x)
        x = nn.Conv2d(128, 128, 3, stride=1, padding=1)(x)
        x = nn.BatchNorm2d(128, 0.8)(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = nn.Upsample(scale_factor=2)(x)
        x = nn.Conv2d(128, 64, 3, stride=1, padding=1)(x)
        x = nn.BatchNorm2d(64, 0.8)(x)
        x = nn.LeakyReLU(0.2, inplace=True)(x)
        x = nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1)(x)
        x = nn.Tanh()(x)
        return x
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = torch.sigmoid(x) # want to predict real or fake so binary output
        return output


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--test_batch_size", type = int, default = 1000, help = "size of the test set batch")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print(args)

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size,
                    'shuffle' : True}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # instantiate discriminator and it's optimizer and loss
    discriminator = Discriminator().to(device)
    D_optimizer = optim.Adadelta(discriminator.parameters(), lr=args.lr)
    D_loss = nn.BCELoss()

    # instantiate generator and it's optiimizer and loss
    generator = Generator(args.latent_dim, args.img_size).to(device)
    G_optimizer = optim.Adadelta(generator.parameters(), lr=args.lr)
    G_loss = nn.BCELoss()

    scheduler = StepLR(D_optimizer, step_size=1, gamma=args.gamma)


    # start training
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, _) in enumerate(train_loader):
            # create valid/fake training labels for discriminator as 1s and 0s
            target_valid = torch.ones((data.shape[0],1), requires_grad=False)
            target_fake = torch.zeros((data.shape[0],1), requires_grad=False)
            data, target_valid, target_fake = data.to(device), target_valid.to(device), target_fake.to(device)


            """
            TRAIN Generator
            """
            G_optimizer.zero_grad()
            z = torch.rand((data.shape[0], args.latent_dim)).to(device)

            gen_imgs = generator(z)  # fake image by the generator
            g_loss = G_loss(discriminator(gen_imgs), target_valid)  # set up loss of the generator such that it is trained to fool discriminator
            g_loss.backward()
            G_optimizer.step()


            """
            TRAIN Discriminator
            """

            # zero out the gradients of the optimizer for this batch
            D_optimizer.zero_grad()
            # forward pass and loss calculation
            d_loss_valid = D_loss(discriminator(data), target_valid)
            d_loss_fake = D_loss(discriminator(gen_imgs.detach()), target_fake)
            d_loss_total = (d_loss_valid + d_loss_fake) / 2
            
            # gradient descent step
            d_loss_total.backward()
            D_optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDiscriminator Loss: {:.6f}\tGenerator Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), d_loss_total.item(), g_loss.item()))
                
                # generate_and_save_images(generator,
                #                  epoch,
                #                  torch.rand((data.shape[0], args.latent_dim)))

                save_image(gen_imgs.data[:10], "images/%d.png" % batch_idx, nrow=5, normalize=True)
        
        # for every epoch print and save a picture
        generate_and_save_images(generator,
                                 epoch,
                                 torch.rand((data.shape[0], args.latent_dim)))
         
        """
        TEST
        """

        scheduler.step()

def generate_and_save_images(model, epoch, test_input):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input)

  fig = plt.figure(figsize=(4,4))
  number_pics_to_show = min(predictions.shape[0], 10)
  for i in range(number_pics_to_show):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, 0, :, :].detach().numpy() * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
        
  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

if __name__ == '__main__':
    main()
