import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.utils import save_image

writer = SummaryWriter()
shutil.rmtree("images")
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = int(np.ceil(opt.img_size / 2 ** 4))
        self.class_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, opt.num_classes), nn.Softmax(dim=1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        classification = self.class_layer(out)

        return classification


# Loss function
classification_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
classifier = ClassifierNet()

if cuda:
    classifier.cuda()
    classification_loss.cuda()
    device = "cuda"
else:
    device = "cpu"

# Initialize weights
classifier.apply(weights_init_normal)

# Configure data loader
os.makedirs("../data", exist_ok=True)

transformations = transforms.Compose(
            [#transforms.Resize(opt.img_size),
            transforms.ToTensor(), 
            transforms.Normalize([0.5], [0.5])
            ])
train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transformations
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

test_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        download=True,
        transform=transformations
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_C = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# ----------
#  Training
# ----------
iteration_counter = 0
labeled_idx = opt.batch_size // 2
for epoch in range(opt.n_epochs):

    for i, (imgs, labels) in enumerate(train_dataloader):

        # create labeled and unlabeled REAL data
        imgs_labeled = imgs

        # put everything on the GPU/CPU
        imgs_labeled, labels = imgs_labeled.to(device), labels.to(device)

        # -----------------
        #  Train Classifier
        # -----------------

        optimizer_C.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        c_loss = classification_loss(classifier(imgs_labeled), labels)

        c_loss.backward()
        optimizer_C.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(train_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            writer.add_scalar("Train/D_loss", c_loss.item(), iteration_counter)
        iteration_counter = iteration_counter + 1
    
    # Testing for classifier accuracy
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            data, target = data.to(device), target.to(device)
            output = classifier(data)
            test_loss += classification_loss(output, target)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    writer.add_scalar("Test/Class_loss", test_loss.item(), iteration_counter)
    writer.add_scalar("Test/Accuracy", correct / len(test_dataloader.dataset), iteration_counter)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))