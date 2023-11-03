import torch.nn as nn
import torch.nn.init as init
import unittest
import numpy as np
import math


def conv_layer(in_planes, out_planes, dropout=False):
    seq = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    if dropout:  # Add dropout module
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)

    return seq


def weights_init(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class Classifier(nn.Module):
    def __init__(self, layer_size, n_classes=0, num_channels=1, dropout=False, image_size=28):
        super(Classifier, self).__init__()

        """
        Builds a CNN to produce embeddings
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param nClasses: If nClasses>0, we want a FC layer at the end with nClasses size.
        :param num_channels: Number of channels of images
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        self.layer1 = conv_layer(num_channels, layer_size, dropout)
        self.layer2 = conv_layer(layer_size, layer_size, dropout)
        self.layer3 = conv_layer(layer_size, layer_size, dropout)
        self.layer4 = conv_layer(layer_size, layer_size, dropout)

        finalSize = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.outSize = finalSize * finalSize * layer_size
        if n_classes > 0:  # We want a linear
            self.useClassification = True
            self.layer5 = nn.Linear(self.outSize, n_classes)
            self.outSize = n_classes
        else:
            self.useClassification = False

        weights_init(self.layer1)
        weights_init(self.layer2)
        weights_init(self.layer3)
        weights_init(self.layer4)

    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        if self.useClassification:
            x = self.layer5(x)
        return x


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward(self):
        pass


if __name__ == '__main__':
    unittest.main()
