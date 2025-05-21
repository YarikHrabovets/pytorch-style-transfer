# 🖼️ Neural Style Transfer Telegram Bot

A telegram bot for image transformation to famous atrists style.

## Background
This project is based on the concept of Neural Style Transfer (NST) — a deep learning technique that blends the content of one image with the artistic style of another. By leveraging the power of convolutional neural networks (CNNs), the model generates new images that preserve the structure of the content image while adopting the visual appearance (colors, textures, brushstrokes) of the style image.

The core idea originates from the groundbreaking research paper “A Neural Algorithm of Artistic Style” by Leon Gatys, which demonstrated that pre-trained CNNs could effectively separate and recombine content and style from different images.

This implementation follows the TensorFlow research notebook [Neural Style Transfer with Eager Execution](https://shorturl.at/aRZCy), showcasing how Eager Execution — a dynamic execution mode in TensorFlow — makes the NST process more intuitive and flexible. It simplifies operations like gradient computation, optimization, and visualization, making it ideal for creative experiments in deep learning.

By reimagining photos in the style of famous paintings, this project highlights the intersection of art and AI, and illustrates how neural networks can be used.

## 🚀 Features
- Upload any photo via Telegram
- Select an artist's style with inline buttons
- High-quality image transformation using TensorFlow
- GPU acceleration support
- Optional Redis integration for scalable callback state handling
- Uses Aiogram 3.x for modern async Telegram bot development

## 🧠 How It Works
1. Loads the pre-trained **VGG19** model for feature extraction.
2. Extracts content features from the user's image.
3. Extracts style features from the selected artist's image.
4. Runs iterative optimization to merge the style into the content.
5. Sends the stylized image back to the user.

## 🛠️ Tech Stack
- [TensorFlow](https://www.tensorflow.org/)
- [Aiogram 3.20](https://docs.aiogram.dev/en/latest/)
- [Pillow (PIL)](https://python-pillow.org/)
- [Redis](https://redis.io/)
- [Python 3.12](https://www.python.org/)

## 🎯 Result

Before processing             |  After processing
:-------------------------:|:-------------------------:
![Before](https://github.com/YarikHrabovets/tensorflow-style-transfer/blob/main/sample/sticker.webp)  |  ![After](https://github.com/YarikHrabovets/tensorflow-style-transfer/blob/main/sample/photo_2025-05-21_10-58-09.jpg)

## 🧪 Setup Instructions
**1. Install redis**

On macOS (with Homebrew)
```sh
brew install redis
redis-server
```

On Ubuntu/Debian
```sh
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis-server.service
sudo systemctl start redis
```

**2. Run this code**
```sh
  git clone https://github.com/YarikHrabovets/tensorflow-style-transfer.git
  cd tensorflow-style-transfer
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
```

**3. Create directory tmp**

**4. Create file .env**
```env
BOT_TOKEN=SET YOUR TELEGRAM BOT TOKEN
HOST=YOUR REDIS HOST
PORT=YOUR REDIS PORT
```

**5. Run the bot**
```sh
python main.py
```
