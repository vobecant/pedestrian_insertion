import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import torchvision.models as models

import os

from models.resnet import ResNet18_OS16, ResNet34_OS16, ResNet50_OS16, ResNet101_OS16, ResNet152_OS16, ResNet18_OS8, ResNet34_OS8
from models.aspp import ASPP, ASPP_Bottleneck

import numpy as np
from PIL import Image


def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1]*(num_blocks - 1) # (stride == 2, num_blocks == 4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion*channels

    layer = nn.Sequential(*blocks) # (*blocks: call with unpacked list entires as arguments)

    return layer

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion*channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(out)) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out

class ResNet_BasicBlock_OS8(nn.Module):
    def __init__(self, num_layers, resnet_path):
        super(ResNet_BasicBlock_OS8, self).__init__()

        if num_layers == 18:
            resnet = models.resnet18()
            # load pretrained model:
            resnet.load_state_dict(torch.load(resnet_path))
            # remove fully connected layer, avg pool, layer4 and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 2
            num_blocks_layer_5 = 2
            print ("pretrained resnet, 18")
        elif num_layers == 34:
            resnet = models.resnet34()
            # load pretrained model:
            resnet.load_state_dict(torch.load("/root/deeplabv3/pretrained_models/resnet/resnet34-333f7ec4.pth"))
            # remove fully connected layer, avg pool, layer4 and layer5:
            self.resnet = nn.Sequential(*list(resnet.children())[:-4])

            num_blocks_layer_4 = 6
            num_blocks_layer_5 = 3
            print ("pretrained resnet, 34")
        else:
            raise Exception("num_layers must be in {18, 34}!")

        self.layer4 = make_layer(BasicBlock, in_channels=128, channels=256, num_blocks=num_blocks_layer_4, stride=1, dilation=2)

        self.layer5 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=num_blocks_layer_5, stride=1, dilation=4)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        # pass x through (parts of) the pretrained ResNet:
        c3 = self.resnet(x) # (shape: (batch_size, 128, h/8, w/8)) (it's called c3 since 8 == 2^3)

        output = self.layer4(c3) # (shape: (batch_size, 256, h/8, w/8))
        output = self.layer5(output) # (shape: (batch_size, 512, h/8, w/8))

        return output


class DeepLabV3(nn.Module):
    def __init__(self, model_id, project_dir, resnet_path):
        super(DeepLabV3, self).__init__()

        self.num_classes = 20

        self.model_id = model_id
        self.project_dir = project_dir
        #self.create_model_dirs()

        self.resnet = ResNet_BasicBlock_OS8(18, resnet_path)
        self.aspp = ASPP(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        h = x.size()[2]
        w = x.size()[3]

        feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        output_orig = F.upsample(output, size=(h, w), mode="nearest") # (shape: (batch_size, num_classes, h, w))

        return output, output_orig

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)


class RoadSegmentation:
    def __init__(self,
                device='cuda:1',
                load_path='/home/tonda/projects/pedestrian_insertion/trained_models/deeplabv3/model_13_2_2_2_epoch_580.pth',
                resnet_path='/home/tonda/projects/pedestrian_insertion/trained_models/deeplabv3/resnet18-5c106cde.pth'):
        self.device = device
        # create and load model
        self.model = DeepLabV3("0", None, resnet_path)
        self.model.load_state_dict(torch.load(load_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.to_tensor = ToTensor()

    def label_img_to_color(self, img):
        label_to_color = {
            0: [128, 64,128],
            1: [244, 35,232],
            2: [ 70, 70, 70],
            3: [102,102,156],
            4: [190,153,153],
            5: [153,153,153],
            6: [250,170, 30],
            7: [220,220,  0],
            8: [107,142, 35],
            9: [152,251,152],
            10: [ 70,130,180],
            11: [220, 20, 60],
            12: [255,  0,  0],
            13: [  0,  0,142],
            14: [  0,  0, 70],
            15: [  0, 60,100],
            16: [  0, 80,100],
            17: [  0,  0,230],
            18: [119, 11, 32],
            19: [81,  0, 81]
            }

        img_height, img_width = img.shape

        img_color = np.zeros((img_height, img_width, 3))
        for row in range(img_height):
            for col in range(img_width):
                label = img[row, col]

                img_color[row, col] = np.array(label_to_color[label])

        return img_color

    def segment(self, image, classes=[0,1], show=False):

        image_t = self.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            segmentation_small, segmentation = self.model(image_t)

        segmentation_small = segmentation_small[0].cpu().detach().numpy()
        pred_label_imgs_small = np.argmax(segmentation_small, axis=0)
        pred_label_imgs_small = pred_label_imgs_small.astype(np.uint8)
        pred_label_img_color_small = self.label_img_to_color(pred_label_imgs_small).astype(np.uint8)

        segmentation = segmentation[0].cpu().detach().numpy()
        pred_label_imgs = np.argmax(segmentation, axis=0)
        pred_label_imgs = pred_label_imgs.astype(np.uint8)

        pred_label_img = pred_label_imgs

        img = np.array(image)
        img = img*np.array([0.229, 0.224, 0.225])
        img = img + np.array([0.485, 0.456, 0.406])
        img = img*255.0
        img = img.astype(np.uint8)

        pred_label_img_color = self.label_img_to_color(pred_label_img)
        overlayed_img = 0.35*img + 0.65*pred_label_img_color
        overlayed_img = overlayed_img.astype(np.uint8)

        overlayed_img_pil = Image.fromarray(overlayed_img)
        if show: overlayed_img_pil.show()

        if classes is None:
            return pred_label_imgs_small
        else:
            segm_classes = np.zeros_like(pred_label_imgs_small, dtype=np.uint8)
            for cid in classes:
                where_class = (pred_label_imgs_small==cid)
                segm_classes[np.where(where_class)] = 1
            assert segm_classes.max()<=1
            return segm_classes


if __name__ == "__main__":
    path = '/home/tonda/projects/pedestrian_insertion/trained_models/deeplabv3/model_13_2_2_2_epoch_580.pth'
    resnet_path = '/home/tonda/projects/pedestrian_insertion/trained_models/deeplabv3/resnet18-5c106cde.pth'
    device = 'cuda:1'
    segmenter = RoadSegmentation(path, device, resnet_path)

    img_path = '/opt/datasets/cityscapes/leftImg8bit/test/berlin/berlin_000001_000019_leftImg8bit.png'
    img = Image.open(img_path)
    segmenter.segment(img, classes=[0,1])