import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



class HyperbolicEmbedder(nn.Module):
    def __init__(self, vocab_size):
        super(HyperbolicEmbedder, self).__init__()

        # the embeddings
        # use polar coordinates
        self.rad1 = nn.Embedding(vocab_size, 1)
        self.theta1 = nn.Embedding(vocab_size, 1)
        self.rad2 = nn.Embedding(vocab_size, 1)
        self.theta2 = nn.Embedding(vocab_size, 1)

        # init weights of radii to be in [0, 1)
        self.rad1.weight.data.uniform_(0, 1)
        self.rad2.weight.data.uniform_(0, 1)

        # init weights for angles to be [0, 2pi)
        self.theta1.weight.data.uniform_(0, 2 * math.pi)
        self.theta2.weight.data.uniform_(0, 2 * math.pi)



    def _atanh(self, x):
        """
        Computes the hyberbolic arctan
        :param x: the vector
        :return: the hyperbolic arctan of the vector
        """
        numer = 1 + x
        denom = 1 - x
        frac = torch.abs(numer / denom)
        factor = torch.log(frac)
        res = 0.5 * factor
        return res

    def get_inner_coord(self, x):
        radius = self.rad1(x)
        theta = self.theta1(x)
        return (radius, theta)

    def get_outer_coord(self, x):
        radius = self.rad2(x)
        theta = self.theta2(x)
        return (radius, theta)

    def _inner_prod(self, x, y):
        """
        Calculates the inner product of two vectors in hyperbolic space
        :param x: the index of the first vector
        :param y: the index of the second vector
        :return: the inner product of the two vectors
        """
        rad1 = self.rad1(x)
        rad2 = self.rad2(y)

        theta1 = self.theta1(x).repeat(1, len(y))
        theta2 = self.theta2(y).view(1, len(y)).repeat(len(x), 1)

        # print(self.theta1(x))
        # print(theta1)
        #
        # print(self.theta2(y))
        # print(theta2)

        diff = torch.cos(theta1 - theta2)

        #print(diff)

        #print(self._atanh(rad1))
        #print(self._atanh(rad2))

        x_factor = self._atanh(rad1).squeeze(0)
        y_factor = (self._atanh(rad2).squeeze(0)).view(1, len(rad2))

        # print(x_factor)
        # print(y_factor)

        res = 4 * (x_factor @ y_factor) * diff
        return res

    def forward(self, x_input, y_target, y_noise):
        positive_prod = self._inner_prod(x_input, y_target)
        positive = torch.sigmoid(positive_prod)
        negatives = self._inner_prod(x_input, y_noise)
        negative_sig = torch.sigmoid(negatives)
        negative = negative_sig.sum()
        return -positive - negative


model = HyperbolicEmbedder(5)
x = torch.LongTensor([1])
y = torch.LongTensor([2])
z = torch.LongTensor([3, 4])
x = Variable(x)
y = Variable(y)
z = Variable(z)
print(model.forward(x, y, z))


#print(model._atanh(torch.Tensor([9, 2, 3])))



