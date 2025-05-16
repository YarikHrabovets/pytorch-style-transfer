import asyncio
import logging
import sys
from os import getenv
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, BufferedInputFile, CallbackQuery
from image_transformation import transform_to_artists_style
from markups import get_artist_keyboard
import uuid
from redis_helper import get_callback_data
import time

load_dotenv()

dp = Dispatcher()
bot = Bot(getenv('BOT_TOKEN'), default=DefaultBotProperties(parse_mode=ParseMode.HTML))


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    answer = f'Hello, {message.from_user.username}!'
    await message.reply(f'{html.quote(answer)}')


@dp.message(F.photo)
async def photo_input_handler(message: Message) -> None:
    photo = await bot.get_file(message.photo[-1].file_id)
    dest = f'./tmp/{uuid.uuid4()}.jpg'
    await bot.download_file(photo.file_path, dest)

    await message.answer(
        text='Image is uploaded. Select an artist:', reply_markup=get_artist_keyboard(dest).as_markup()
    )


@dp.message(~F.photo)
async def text_input_handler(message: Message) -> None:
    answer = html.bold('Unsupported input type.\nYou should upload an image')
    await message.answer(text=answer)


async def send_progress(callback: CallbackQuery, iteration, loss, start_time) -> None:
    curr = time.time()
    await callback.message.edit_text(
        f'Iteration {iteration}/100 with loss: {loss:.2f}\nRunning for {(curr - start_time):.0f} seconds...',
        reply_markup=None
    )


@dp.callback_query(F.data.len() == 8)
async def handle_artist_callback(callback: CallbackQuery):
    data = get_callback_data(callback.data)
    if data is None:
        await callback.answer('This option has expired.', show_alert=True)
        return

    artist_key, artist_name, filename = data.values()
    start = time.time()
    result_buffer = await transform_to_artists_style(
        filename, artist_key, lambda i, loss: send_progress(callback, i, loss, start)
    )
    file = BufferedInputFile(result_buffer, filename='stylized.jpg')
    await callback.message.answer_photo(photo=file, caption=f'Image in {artist_name} style')


async def main() -> None:
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
