from neural_style_transfer import NeuralStyleTransfer
from tensorflow import keras
from PIL import Image
from io import BytesIO

vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

style_outputs = [vgg.get_layer(name).output for name in style_layers]
content_outputs = [vgg.get_layer(name).output for name in content_layers]
model_outputs = style_outputs + content_outputs

model = keras.models.Model(vgg.input, model_outputs)
for layer in model.layers:
    layer.trainable = False


async def transform_to_van_gogh_style(data, progress_callback):
    image_input = Image.open(data).convert('RGB')
    image_style = Image.open('./styles/Vincent-van-Gogh.jpg').convert('RGB')
    neural_transfer = NeuralStyleTransfer(image_input, image_style, model, num_style_layers, num_content_layers)
    best_img, best_loss = await neural_transfer.process_iterations(progress_callback)

    image = Image.fromarray(best_img.astype('uint8'), 'RGB')
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)

    return buffer.getvalue()

