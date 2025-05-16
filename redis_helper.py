import redis
import uuid
import json
from os import getenv
from dotenv import load_dotenv

load_dotenv()
r = redis.Redis(host=getenv('HOST'), port=int(getenv('PORT')), decode_responses=True)


def set_callback_data(data: dict, ttl: int = 300) -> str:
    token = uuid.uuid4().hex[:8]
    r.set(f'cb:{token}', json.dumps(data), ex=ttl)
    return token


def get_callback_data(token: str) -> dict | None:
    raw = r.get(f'cb:{token}')
    return json.loads(raw) if raw else None
