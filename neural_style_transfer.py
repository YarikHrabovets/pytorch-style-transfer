import numpy as np
import tensorflow as tf
from tensorflow import keras


class NeuralStyleTransfer:
    num_iterations = 100
    content_weight = 1e3
    style_weight = 1e-2
    iter_count = 1

    def __init__(self, img, style, model, num_style_layers, num_content_layers):
        self.img = img
        self.style = style
        self.model = model
        self.num_style_layers = num_style_layers
        self.num_content_layers = num_content_layers
        self.x_img = keras.applications.vgg19.preprocess_input(np.expand_dims(self.img, axis=0))
        self.x_style = keras.applications.vgg19.preprocess_input(np.expand_dims(self.style, axis=0))
        self.style_features, self.content_features = self.get_feature_representations()
        self.gram_style_features = [gram_matrix(style_feature) for style_feature in self.style_features]
        self.init_image = np.copy(self.x_img)
        self.init_image = tf.Variable(self.init_image, dtype=tf.float32)
        self.opt = tf.compat.v1.train.AdamOptimizer(learning_rate=2, beta1=0.99, epsilon=1e-1)
        self.best_loss, self.best_img = float('inf'), None
        self.loss_weights = (NeuralStyleTransfer.style_weight, NeuralStyleTransfer.content_weight)
        self.cfg = {
            'model': self.model,
            'loss_weights': self.loss_weights,
            'init_image': self.init_image,
            'gram_style_features': self.gram_style_features,
            'content_features': self.content_features,
            'num_style_layers': self.num_style_layers,
            'num_content_layers': self.num_content_layers
        }

    def get_feature_representations(self):
        style_features = [style_layer[0] for style_layer in self.model(self.x_style)[:self.num_style_layers]]
        content_features = [content_layer[0] for content_layer in self.model(self.x_img)[self.num_style_layers:]]
        return style_features, content_features

    async def process_iterations(self, progress_callback=None):
        norm_means = np.array([103.939, 116.779, 123.68], dtype=np.float32)
        min_vals = tf.constant(-norm_means)
        max_vals = tf.constant(255 - norm_means)
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

        with tf.device(device):
            for i in range(NeuralStyleTransfer.num_iterations):
                with tf.GradientTape() as tape:
                    all_loss = compute_loss(**self.cfg)

                loss, style_score, content_score = all_loss
                grads = tape.gradient(loss, self.init_image)
                self.opt.apply_gradients([(grads, self.init_image)])
                clipped = tf.clip_by_value(self.init_image, min_vals, max_vals)
                self.init_image.assign(clipped)
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best_img = deprocess_img(self.init_image.numpy())

                    if progress_callback:
                        await progress_callback(i + 1, loss)

        return self.best_img, self.best_loss


def deprocess_img(img):
    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)

    if len(x.shape) != 3:
        return 'Invalid input to deprocessing image'

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))


def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)


def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target))


def compute_loss(model, loss_weights, init_image, gram_style_features,
                 content_features, num_style_layers, num_content_layers):

    style_weight, content_weight = loss_weights

    model_outputs = model(init_image)

    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]

    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)

    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score
    return loss, style_score, content_score
