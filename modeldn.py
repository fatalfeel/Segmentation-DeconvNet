import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

# refer to:
# http://cvlab.postech.ac.kr/research/deconvnet/
# http://cvlab.postech.ac.kr/research/deconvnet/model/DeconvNet/DeconvNet_inference_deploy.prototxt
class Conv_Deconv(nn.Module):
    def __init__(self):
        super(Conv_Deconv,self).__init__()
        self.name       = 'DeconvNet'

        # conv1
        self.conv1_1    = torch.nn.Conv2d(3, 16,    4, 1, 0)
        self.bn1_1      = torch.nn.BatchNorm2d(16)
        self.relu1_1    = torch.nn.ReLU()
        self.maxpool1   = torch.nn.MaxPool2d(2, return_indices=True)

        # conv2
        self.conv2_1    = torch.nn.Conv2d(16, 32,   5, 1, 0)
        self.bn2_1      = torch.nn.BatchNorm2d(32)
        self.relu2_1    = torch.nn.ReLU()
        self.maxpool2   = torch.nn.MaxPool2d(2, return_indices=True)

        # conv3
        self.conv3_1    = torch.nn.Conv2d(32, 64,   3, 1, 0)
        self.bn3_1      = torch.nn.BatchNorm2d(64)
        self.relu3_1    = torch.nn.ReLU()

############################deconv##########################################
        self.deconv3_1  = torch.nn.ConvTranspose2d(64, 32,      3, 1 ,0)
        self.debn3_1    = torch.nn.BatchNorm2d(32)
        self.derelu3_1  = torch.nn.ReLU()

        self.maxunpool2 = torch.nn.MaxUnpool2d(kernel_size=2)
        self.deconv2_1  = torch.nn.ConvTranspose2d(32, 16,      5, 1, 0)
        self.debn2_1    = torch.nn.BatchNorm2d(16)
        self.derelu2_1  = torch.nn.ReLU()

        self.maxunpool1 = torch.nn.MaxUnpool2d(kernel_size=2)
        self.deconv1_1  = torch.nn.ConvTranspose2d(16, 3,       4, 1, 0)
        self.debn1_1    = torch.nn.BatchNorm2d(3)
        self.derelu1_1  = torch.nn.ReLU()

        self.deconv_class   = torch.nn.ConvTranspose2d(3, 21,   1, 1, 0)
        self.sig            = nn.Sigmoid()

        self.celoss_fn      = torch.nn.CrossEntropyLoss()  # NEVER FORGET TO CHANGE LOSS_NAME WHEN CHANGING THE LOSS

    # normalize filter values to 0-1 so we can visualize them
    def NormalizeImg(self, img):
        nimg = (img - img.min()) / (img.max() - img.min())
        return nimg

    def show_MNIST(self, img):
        grid    = torchvision.utils.make_grid(img)
        trimg   = grid.detach().numpy().transpose(1, 2, 0)
        plt.imshow(trimg)
        plt.title('Batch from dataloader')
        plt.axis('off')
        plt.show()

    def forward(self, data):
        #self.show_MNIST(data[0])
        output          = self.conv1_1(data)
        output          = self.bn1_1(output)
        output          = self.relu1_1(output)
        size1_1         = output.size()
        output, indices1= self.maxpool1(output)

        output          = self.conv2_1(output)
        output          = self.bn2_1(output)
        output          = self.relu2_1(output)
        size2_1         = output.size()
        output, indices2= self.maxpool2(output)

        output          = self.conv3_1(output)
        output          = self.bn3_1(output)
        output          = self.relu3_1(output)
###################################################################################
        output = self.deconv3_1(output)
        output = self.debn3_1(output)
        output = self.derelu3_1(output)

        output = self.maxunpool2(output, indices2, size2_1)
        output = self.deconv2_1(output)
        output = self.debn2_1(output)
        output = self.derelu2_1(output)

        output = self.maxunpool1(output, indices1, size1_1)
        output = self.deconv1_1(output)
        output = self.debn1_1(output)
        output = self.derelu1_1(output)

        score   = self.deconv_class(output)
        score   = self.sig(score)
        #nimg = self.NormalizeImg(output[0])
        #self.show_MNIST(nimg)

        return output, score

    def lossfunction(self, output, label):
        return self.celoss_fn(output, label)
