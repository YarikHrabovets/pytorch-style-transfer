import asyncio
import logging
import sys
from os import getenv
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, KeyboardButton
from aiogram.utils import keyboard

load_dotenv()

dp = Dispatcher()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    answer = f'Hello, {message.from_user.username}!'
    await message.reply(f'{html.quote(answer)}')


@dp.message()
async def keyboard_handler(message: Message) -> None:
    if message.photo:
        await message.send_copy(chat_id=message.chat.id)
    else:
        await message.answer('You should upload an image')


async def main() -> None:
    bot = Bot(getenv('BOT_TOKEN'), default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
