from utils import ImageTransformerNet
import torch
from utils import load_image
from io import BytesIO
from torchvision.transforms import ToPILImage


def transform_to_van_gogh_style(image: BytesIO) -> BytesIO:
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = ImageTransformerNet().to(device)
    model.load_state_dict(torch.load('./saved_models/VincentVanGogh_model.pth'))
    model.eval()

    img = load_image(image, device=device)

    with torch.no_grad():
        output = model(img)

    output = output.clamp(0, 255) / 255.0
    to_pil = ToPILImage()
    image = to_pil(output.squeeze(0))
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    return buffer
