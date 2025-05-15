from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from PIL import Image
from utils import normalize, gram_matrix, CustomImageDataset, VGGFeatures, ImageTransformerNet
import time

dataset = CustomImageDataset('./images/content')

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
transformer = ImageTransformerNet().to(device)
vgg = VGGFeatures(device)

with Image.open('./images/styles/Vincent-van-Gogh.jpg') as img:
    style_image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])(img).unsqueeze(0).to(device)

optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-3)
style_weight = 1e5

mse = torch.nn.MSELoss()


def start_train():
    for i, batch in enumerate(dataloader):
        if i == 100:
            print(f'Time for 100 steps: {time.time() - start:.2f}s')
            break

        content_images = batch.to(device)

        generated = transformer(content_images)

        gen_features = vgg(normalize(generated))
        content_features = vgg(normalize(content_images))
        style_resized = torch.nn.functional.interpolate(
            style_image, size=generated.shape[2:], mode='bilinear', align_corners=False
        )
        style_features = vgg(normalize(style_resized))

        if gen_features[-1].shape != content_features[-1].shape:
            content_features[-1] = torch.nn.functional.interpolate(
                content_features[-1], size=gen_features[-1].shape[2:], mode='bilinear', align_corners=False
            )

        content_loss = mse(gen_features[-1], content_features[-1])

        style_loss = 0
        for gf, sf in zip(gen_features, style_features):
            style_loss += mse(gram_matrix(gf), gram_matrix(sf))

        total_loss = content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    torch.save(transformer.state_dict(), './saved_models/VincentVanGogh_model.pth')


if __name__ == '__main__':
    start = time.time()
    start_train()
