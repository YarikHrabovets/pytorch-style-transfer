import os
import uuid
import json
from os import getenv
from dotenv import load_dotenv
from redis.asyncio import Redis

load_dotenv()

r = Redis(host=getenv('HOST'), port=int(getenv('PORT')), decode_responses=True)


async def set_callback_data(data: dict, ttl: int = 300) -> str:
    token = uuid.uuid4().hex[:8]
    await r.set(f'cb:{token}', json.dumps(data), ex=ttl)
    await r.set(f'shadow:{token}', json.dumps(data.get('file_path')))
    return token


async def get_callback_data(prefix: str, token: str) -> dict | str | None:
    raw = await r.get(f'{prefix}:{token}')
    return json.loads(raw) if raw else None


async def expired_keys_listener():
    pubsub = r.pubsub()
    await pubsub.psubscribe('__keyevent@0__:expired')
    async for message in pubsub.listen():
        if message.get('type') == 'pmessage':
            token = message.get('data')
            if token is None:
                continue

            token = token.split(':')[1]
            data = await get_callback_data(prefix='shadow', token=token)
            if data is None:
                continue
            try:
                os.remove(data)
                await r.delete(f'shadow:{token}')
            except FileNotFoundError:
                print(f'File not found: {data}')
