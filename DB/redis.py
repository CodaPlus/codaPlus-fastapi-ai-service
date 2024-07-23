from langchain.cache import RedisCache
from redis import Redis
import os
from dotenv import load_dotenv
load_dotenv("../.env") 

redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", 6379)

REDIS_VECTOR_DB = f"redis://{redis_host}:{redis_port}/0"

try: 
    r = Redis(
        host=redis_host,
        port=redis_port,
        db=os.getenv("REDIS_DB", 6),
    )
except Exception as e:
    print("An unexpected error occurred:", e)


try: 
    redis_cache = RedisCache(redis_=Redis(
        host=redis_host,
        port=redis_port,
        db=int(os.getenv("REDIS_CACHE_DB", 7)),
    ), ttl=600) # 10 minutes
except Exception as e:
    print("An unexpected error occurred:", e)
