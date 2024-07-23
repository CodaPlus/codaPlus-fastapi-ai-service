import chromadb
import threading
from dotenv import load_dotenv
load_dotenv("../.env") 
import os 


class ChromaClientSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ChromaClientSingleton, cls).__new__(cls)
            return cls._instance

    def __init__(self):
        with self.__class__._lock:
            # This ensures that initialization happens only once
            if not hasattr(self, 'is_initialized'):
                self._initialize_chroma()
                self.is_initialized = True

    def _initialize_chroma(self):
        self.client = chromadb.HttpClient(
            host=os.getenv("CHROMA_HOST", "127.0.0.1"),
            port=int(os.getenv("CHROMA_PORT", 6000)),
            settings=chromadb.Settings(allow_reset=True),
            ssl=bool(os.getenv("CHROMA_DB_SSL_ENABLED", False))
        )
    
chroma_client = ChromaClientSingleton()