import asyncio
import logging
import sys
from os import getenv
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, html, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, KeyboardButton, BufferedInputFile
from aiogram.utils import keyboard
from io import BytesIO
from image_transformation import transform_to_van_gogh_style

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
    buffer = BytesIO()
    await bot.download_file(photo.file_path, buffer)
    buffer.seek(0)

    sent = await message.answer(text='Image is uploaded. Starting transformation...')
    result_buffer = await transform_to_van_gogh_style(buffer, lambda i, loss: send_progress(sent, i, loss))
    file = BufferedInputFile(result_buffer, filename='stylized.jpg')
    await message.answer_photo(photo=file, caption='Image in Van Gogh style')


@dp.message(~F.photo)
async def text_input_handler(message: Message) -> None:
    answer = html.bold('Unsupported input type.\nYou should upload an image')
    await message.answer(text=answer)


async def send_progress(sent: Message, iteration, loss) -> None:
    await sent.edit_text(f'Iteration {iteration}/100 with loss: {loss:.2f}')


async def main() -> None:
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
