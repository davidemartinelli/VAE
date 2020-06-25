import argparse

parser = argparse.ArgumentParser()

#training
parser.add_argument('-e', type=int, default=20, help='Number of training epochs. Default: 20')
parser.add_argument('-b', type=int, default=128, help='Batch size. Default: 128')
parser.add_argument('-L', type=int, default=10, help='Number of samples per datapoint during optimization. Default: 10')
parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate. Default: 1e-3')

#model
parser.add_argument('-hl', type=int, default=1, help='Number of hidden layers. Default: 1')
parser.add_argument('-hu', type=int, default=500, help='Number of hidden units. Default: 500')
parser.add_argument('-D', type=int, default=20, help='Dimensionality of latent space. Default: 20')

args = parser.parse_args()