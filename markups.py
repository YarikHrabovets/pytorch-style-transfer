import os
from aiogram.utils.keyboard import InlineKeyboardBuilder
from redis_helper import set_callback_data


async def get_artist_keyboard(file_path) -> InlineKeyboardBuilder:
    artists = [file.split('.')[0].replace('-', ' ') for file in os.listdir('./styles')]
    builder = InlineKeyboardBuilder()
    for i, artist in enumerate(artists):
        token = await set_callback_data({
            'artist_key': i,
            'artist_name': artist,
            'file_path': file_path
        })
        builder.button(text=artist, callback_data=token)
    builder.adjust(2)
    return builder
