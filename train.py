import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

channels = 1
noise_dim = 100
lr = 0.0002
batch_size = 128
epochs = 50
features = 64
image_size = 64


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

loader = DataLoader(MNIST(root="MNIST/", download=True, train=True, transform=transform), batch_size=batch_size, shuffle=True)

generator = Generator(noise_dim=noise_dim, channels=channels, features=features)
discriminator = Discriminator(in_channels=channels, features=features)
initialize_weights(generator)
initialize_weights(discriminator)

optim_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optim_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))

loss = nn.BCELoss()

fixed_noise = torch.randn(32, noise_dim, 1, 1)
writer_fake = SummaryWriter("./runs/fake")
writer_real = SummaryWriter("./runs/real")
steps = 0

for epoch in range(epochs):
    for batch_idx, (real, _) in tqdm(enumerate(loader)):

        noise = torch.randn((batch_size, noise_dim, 1, 1))
        fake = generator(noise)

        disc_real = discriminator(real).view(-1)
        disc_fake = discriminator(fake).view(-1)

        lossD_real = loss(disc_real, torch.ones_like(disc_real))
        lossD_fake = loss(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2

        discriminator.zero_grad()
        lossD.backward(retain_graph=True)
        optim_disc.step()

        output = discriminator(fake).view(-1)
        lossG = loss(output, torch.ones_like(output))

        generator.zero_grad()
        lossG.backward()
        optim_gen.step()

    with torch.no_grad():
        fake = generator(fixed_noise)
        fake = np.array(fake.cpu())
        fake = np.squeeze(fake, axis=1)
        print(fake.shape)
        fig = plt.figure(figsize =(10, 10))

        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.imshow(fake[i], cmap="binary")
            plt.axis('off')
        plt.show()
