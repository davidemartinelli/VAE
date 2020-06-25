import torch
from torch.utils.data import DataLoader

import torchvision

from model import VAE
from args import args
from utils import BinaryMNIST

import time
import os

IMAGES_FOLDER = 'images'
N_INPUT = 784 #dimensionality images
MANIFOLD_WIDTH = 20 #only used if D=2

train_loader = DataLoader(BinaryMNIST(), batch_size=args.b, shuffle=True, num_workers=1, pin_memory=True)
val_loader = DataLoader(BinaryMNIST(train=False), batch_size=args.b, shuffle=False, num_workers=1, pin_memory=True)

model = VAE(N_INPUT, args.D, args.hu, args.hl)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

start = time.time()

for e in range(args.e):
    print(f'Epoch {e + 1}...')

    #training
    for images, labels in train_loader:
        images = images.to(model.device)
        loss = 0

        for l in range(args.L):
            mu, std, out = model(images)

            #computing the loss
            kl = ((1 + torch.log(std ** 2) - mu ** 2 - std ** 2) / 2).sum()
            log_prob = (images * torch.log(out + 1e-7) + (1 - images) * torch.log(1 - out + 1e-7)).sum()
            loss += - (kl + log_prob) / args.L

        loss *= len(train_loader) / args.b 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #validation
    with torch.no_grad():
        loss = 0
        for images, labels in val_loader:
            images = images.to(model.device)

            mu, std, out = model(images)

            kl = ((1 + torch.log(std ** 2) - mu ** 2 - std ** 2) / 2).sum()
            log_prob = (images * torch.log(out + 1e-7) + (1 - images) * torch.log(1 - out + 1e-7)).sum()
            loss += - (kl + log_prob) 

    print('Average validation loss:', loss.item() / len(val_loader))

    end = time.time()
    print(f'Done in {round(end - start)} seconds')
    start = end

print('Generating images from the model... ', end='')
samples = model.sample(n_samples=64)

if not os.path.exists(IMAGES_FOLDER):
            os.mkdir(IMAGES_FOLDER)

torchvision.utils.save_image(samples, os.path.join(IMAGES_FOLDER, f'samples_hl:{args.hl}_hu:{args.hu}_D:{args.D}_b:{args.b}_L:{args.L}_e:{args.e}.png'), nrow=8, padding=0)

print('Done!')

if args.D == 2:
    print('Producing the learned data manifold... ', end='')
    manifold = model.generate_manifold(MANIFOLD_WIDTH)
    torchvision.utils.save_image(manifold, os.path.join(IMAGES_FOLDER, f'manifold_hl:{args.hl}_hu:{args.hu}_D:{args.D}_b:{args.b}_L:{args.L}_e:{args.e}.png'), nrow=MANIFOLD_WIDTH, padding=0)

    print('Done!')