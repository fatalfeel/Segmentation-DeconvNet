import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

# refer to:
# http://cvlab.postech.ac.kr/research/deconvnet/
# http://cvlab.postech.ac.kr/research/deconvnet/model/DeconvNet/DeconvNet_inference_deploy.prototxt
class Conv_Deconv(nn.Module):
    def __init__(self, fullwork):
        super(Conv_Deconv,self).__init__()
        self.name       = 'DeconvNet'
        self.fullwork   = fullwork

        # conv1
        self.conv1_1    = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1      = torch.nn.BatchNorm2d(64)
        self.relu1_1    = torch.nn.ReLU()
        if self.fullwork is True:
            self.conv1_2    = torch.nn.Conv2d(64, 64, 3, padding=1)
            self.bn1_2      = torch.nn.BatchNorm2d(64)
            self.relu1_2    = torch.nn.ReLU()
        self.maxpool1   = torch.nn.MaxPool2d(2, stride=2, return_indices=True)

        # conv2
        self.conv2_1    = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1      = torch.nn.BatchNorm2d(128)
        self.relu2_1    = torch.nn.ReLU()
        if self.fullwork is True:
            self.conv2_2    = torch.nn.Conv2d(128, 128, 3, padding=1)
            self.bn2_2      = torch.nn.BatchNorm2d(128)
            self.relu2_2    = torch.nn.ReLU()
        self.maxpool2   = torch.nn.MaxPool2d(2, stride=2, return_indices=True)

        # conv3
        self.conv3_1    = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1      = torch.nn.BatchNorm2d(256)
        self.relu3_1    = torch.nn.ReLU()
        if self.fullwork is True:
            self.conv3_2    = torch.nn.Conv2d(256, 256, 3, padding=1)
            self.bn3_2      = torch.nn.BatchNorm2d(256)
            self.relu3_2    = torch.nn.ReLU()
            self.conv3_3    = torch.nn.Conv2d(256, 256, 3, padding=1)
            self.bn3_3      = torch.nn.BatchNorm2d(256)
            self.relu3_3    = torch.nn.ReLU()
        self.maxpool3   = torch.nn.MaxPool2d(2, stride=2, return_indices=True)

        # conv4
        self.conv4_1    = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1      = torch.nn.BatchNorm2d(512)
        self.relu4_1    = torch.nn.ReLU()
        if self.fullwork is True:
            self.conv4_2    = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.bn4_2      = torch.nn.BatchNorm2d(512)
            self.relu4_2    = torch.nn.ReLU()
            self.conv4_3    = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.bn4_3      = torch.nn.BatchNorm2d(512)
            self.relu4_3    = torch.nn.ReLU()
        self.maxpool4  = torch.nn.MaxPool2d(2, stride=2, return_indices=True)

        # conv5
        self.conv5_1    = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn5_1      = torch.nn.BatchNorm2d(512)
        self.relu5_1    = torch.nn.ReLU()
        if self.fullwork is True:
            self.conv5_2    = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.bn5_2      = torch.nn.BatchNorm2d(512)
            self.relu5_2    = torch.nn.ReLU()
            self.conv5_3    = torch.nn.Conv2d(512, 512, 3, padding=1)
            self.bn5_3      = torch.nn.BatchNorm2d(512)
            self.relu5_3    = torch.nn.ReLU()
        self.maxpool5   = torch.nn.MaxPool2d(2, stride=2, return_indices=True)

        if self.fullwork is True:
            self.fc6    = torch.nn.Conv2d( 512, 4096, 7)
            self.bn6    = torch.nn.BatchNorm2d(4096)
        else:
            self.fc6 = torch.nn.Conv2d(512, 512, 7)
            self.bn6 = torch.nn.BatchNorm2d(512)

        self.relu6  = torch.nn.ReLU()

        if self.fullwork is True:
            self.fc7    = torch.nn.Conv2d(4096, 4096, 1)
            self.bn7    = torch.nn.BatchNorm2d(4096)
        else:
            self.fc7 = torch.nn.Conv2d(512, 512, 1)
            self.bn7 = torch.nn.BatchNorm2d(512)

        self.relu7  = torch.nn.ReLU()

############################deconv##########################################
        if self.fullwork is True:
            self.defc6 = torch.nn.ConvTranspose2d(4096, 512, 7)
        else:
            self.defc6 = torch.nn.ConvTranspose2d(512, 512, 7)

        self.debn6      = torch.nn.BatchNorm2d(512)
        self.derelu6    = torch.nn.ReLU()

        self.maxunpool5 = torch.nn.MaxUnpool2d(2, stride=2)
        if self.fullwork is True:
            self.deconv5_3  = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
            self.debn5_3    = torch.nn.BatchNorm2d(512)
            self.derelu5_3  = torch.nn.ReLU()
            self.deconv5_2  = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
            self.debn5_2    = torch.nn.BatchNorm2d(512)
            self.derelu5_2  = torch.nn.ReLU()
        self.deconv5_1  = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
        self.debn5_1    = torch.nn.BatchNorm2d(512)
        self.derelu5_1  = torch.nn.ReLU()

        self.maxunpool4 = torch.nn.MaxUnpool2d(2, stride=2)
        if self.fullwork is True:
            self.deconv4_3  = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
            self.debn4_3    = torch.nn.BatchNorm2d(512)
            self.derelu4_3  = torch.nn.ReLU()
            self.deconv4_2  = torch.nn.ConvTranspose2d(512, 512, 3, padding=1)
            self.debn4_2    = torch.nn.BatchNorm2d(512)
            self.derelu4_2  = torch.nn.ReLU()
        self.deconv4_1  = torch.nn.ConvTranspose2d(512, 256, 3, padding=1)
        self.debn4_1    = torch.nn.BatchNorm2d(256)
        self.derelu4_1  = torch.nn.ReLU()

        self.maxunpool3 = torch.nn.MaxUnpool2d(2, stride=2)
        if self.fullwork is True:
            self.deconv3_3  = torch.nn.ConvTranspose2d(256, 256, 3, padding=1)
            self.debn3_3    = torch.nn.BatchNorm2d(256)
            self.derelu3_3  = torch.nn.ReLU()
            self.deconv3_2  = torch.nn.ConvTranspose2d(256, 256, 3, padding=1)
            self.debn3_2    = torch.nn.BatchNorm2d(256)
            self.derelu3_2  = torch.nn.ReLU()
        self.deconv3_1  = torch.nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.debn3_1    = torch.nn.BatchNorm2d(128)
        self.derelu3_1  = torch.nn.ReLU()

        self.maxunpool2 = torch.nn.MaxUnpool2d(2, stride=2)
        if self.fullwork is True:
            self.deconv2_2  = torch.nn.ConvTranspose2d(128, 128, 3, padding=1)
            self.debn2_2    = torch.nn.BatchNorm2d(128)
            self.derelu2_2  = torch.nn.ReLU()
        self.deconv2_1  = torch.nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.debn2_1    = torch.nn.BatchNorm2d(64)
        self.derelu2_1  = torch.nn.ReLU()

        self.maxunpool1 = torch.nn.MaxUnpool2d(2, stride=2)
        if self.fullwork is True:
            self.deconv1_2  = torch.nn.ConvTranspose2d(64, 64, 3, padding=1)
            self.debn1_2    = torch.nn.BatchNorm2d(64)
            self.derelu1_2  = torch.nn.ReLU()
        self.deconv1_1  = torch.nn.ConvTranspose2d(64, 3, 3, padding=1)
        self.debn1_1    = torch.nn.BatchNorm2d(3)
        self.derelu1_1  = torch.nn.ReLU()

        self.deconv_score = torch.nn.ConvTranspose2d(3, 21, 1)

        self.loss_name  = 'CrossEntropyLoss'
        self.celoss_fn  = torch.nn.CrossEntropyLoss()  # NEVER FORGET TO CHANGE LOSS_NAME WHEN CHANGING THE LOSS

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
        if self.fullwork is True:
            output          = self.conv1_2(output)
            output          = self.bn1_2(output)
            output          = self.relu1_2(output)
            size1_2         = output.size()
        output, indices1= self.maxpool1(output)

        output          = self.conv2_1(output)
        output          = self.bn2_1(output)
        output          = self.relu2_1(output)
        size2_1         = output.size()
        if self.fullwork is True:
            output          = self.conv2_2(output)
            output          = self.bn2_2(output)
            output          = self.relu2_2(output)
            size2_2         = output.size()
        output, indices2= self.maxpool2(output)

        output          = self.conv3_1(output)
        output          = self.bn3_1(output)
        output          = self.relu3_1(output)
        size3_1         = output.size()
        if self.fullwork is True:
            output          = self.conv3_2(output)
            output          = self.bn3_2(output)
            output          = self.relu3_2(output)
            size3_2         = output.size()
            output          = self.conv3_3(output)
            output          = self.bn3_3(output)
            output          = self.relu3_3(output)
            size3_3         = output.size()
        output, indices3= self.maxpool3(output)

        output          = self.conv4_1(output)
        output          = self.bn4_1(output)
        output          = self.relu4_1(output)
        size4_1         = output.size()
        if self.fullwork is True:
            output          = self.conv4_2(output)
            output          = self.bn4_2(output)
            output          = self.relu4_2(output)
            size4_2         = output.size()
            output          = self.conv4_3(output)
            output          = self.bn4_3(output)
            output          = self.relu4_3(output)
            size4_3         = output.size()
        output, indices4= self.maxpool4(output)

        output          = self.conv5_1(output)
        output          = self.bn5_1(output)
        output          = self.relu5_1(output)
        size5_1         = output.size()
        if self.fullwork is True:
            output          = self.conv5_2(output)
            output          = self.bn5_2(output)
            output          = self.relu5_2(output)
            size5_2         = output.size()
            output          = self.conv5_3(output)
            output          = self.bn5_3(output)
            output          = self.relu5_3(output)
            size5_3         = output.size()
        output, indices5= self.maxpool5(output)

        output          = self.fc6(output)
        output          = self.bn6(output)
        output          = self.relu6(output)
        output          = self.fc7(output)
        output          = self.bn7(output)
        output          = self.relu7(output)

###################################################################################
        output = self.defc6(output)
        output = self.debn6(output)
        output = self.derelu6(output)

        output = self.maxunpool5(output, indices5, size5_1)
        if self.fullwork is True:
            output = self.deconv5_3(output)
            output = self.debn5_3(output)
            output = self.derelu5_3(output)
            output = self.deconv5_2(output)
            output = self.debn5_2(output)
            output = self.derelu5_2(output)
        output = self.deconv5_1(output)
        output = self.debn5_1(output)
        output = self.derelu5_1(output)

        output = self.maxunpool4(output, indices4, size4_1)
        if self.fullwork is True:
            output = self.deconv4_3(output)
            output = self.debn4_3(output)
            output = self.derelu4_3(output)
            output = self.deconv4_2(output)
            output = self.debn4_2(output)
            output = self.derelu4_2(output)
        output = self.deconv4_1(output)
        output = self.debn4_1(output)
        output = self.derelu4_1(output)

        output = self.maxunpool3(output, indices3, size3_1)
        if self.fullwork is True:
            output = self.deconv3_3(output)
            output = self.debn3_3(output)
            output = self.derelu3_3(output)
            output = self.deconv3_2(output)
            output = self.debn3_2(output)
            output = self.derelu3_2(output)
        output = self.deconv3_1(output)
        output = self.debn3_1(output)
        output = self.derelu3_1(output)

        output = self.maxunpool2(output, indices2, size2_1)
        if self.fullwork is True:
            output = self.deconv2_2(output)
            output = self.debn2_2(output)
            output = self.derelu2_2(output)
        output = self.deconv2_1(output)
        output = self.debn2_1(output)
        output = self.derelu2_1(output)

        output = self.maxunpool1(output, indices1, size1_1)
        if self.fullwork is True:
            output = self.deconv1_2(output)
            output = self.debn1_2(output)
            output = self.derelu1_2(output)
        output = self.deconv1_1(output)
        output = self.debn1_1(output)
        output = self.derelu1_1(output)

        score  = self.deconv_score(output)

        #nimg = self.NormalizeImg(output[0])
        #self.show_MNIST(nimg)

        return output, score

    def lossfunction(self, output, label):
        return self.celoss_fn(output, label)
