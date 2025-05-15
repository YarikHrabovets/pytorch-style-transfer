import torch.nn as nn
from torch import tensor, bmm
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import os


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        reflection_padding = kernel_size // 2
        super().__init__(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU()
        )


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = ConvLayer(in_c, out_c, kernel_size, stride)

    def forward(self, x):
        x = self.upsample(x)
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True)
        )

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class ImageTransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvLayer(3, 32, 9, 1),
            ConvLayer(32, 64, 3, 2),
            ConvLayer(64, 128, 3, 2),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])
        self.decoder = nn.Sequential(
            UpsampleConvLayer(128, 64, 3, 1),
            UpsampleConvLayer(64, 32, 3, 1),
            nn.Conv2d(32, 3, 9, 1),
            nn.Tanh()
        )

    def forward(self, x) -> nn.Sequential:
        x = self.encoder(x)
        x = self.residuals(x)
        return self.decoder(x)


class CustomImageDataset(Dataset):
    def __init__(self, content_dir):
        self.content_dir = content_dir
        self.files = os.listdir(self.content_dir)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.content_dir, self.files[idx])).convert('RGB')
        return self.transform(image)


class VGGFeatures(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        self.layers = nn.Sequential(*list(vgg.children())[:21])

        self.feature_map = {
            'relu1_1': 1,
            'relu2_1': 6,
            'relu3_1': 11,
            'relu4_1': 20,
            'relu4_2': 21
        }

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.feature_map.values():
                features.append(x)
        return features


def normalize(t):
    mean = tensor([0.485, 0.456, 0.406]).to(t.device).view(1, 3, 1, 1)
    std = tensor([0.229, 0.224, 0.225]).to(t.device).view(1, 3, 1, 1)
    return (t / 255.0 - mean) / std


def gram_matrix(feat):
    (b, c, h, w) = feat.size()
    feat = feat.view(b, c, h * w)
    gram = bmm(feat, feat.transpose(1, 2))
    return gram / (c * h * w)


def load_image(image_path, size=(256, 256), device='cpu'):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image = transform(image).unsqueeze(0).to(device)
    return image
