from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders import (
    YoutubeLoader,
    WebBaseLoader,
    PyPDFLoader,
    JSONLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from .utils import UnstructuredExcelLoader, is_youtube_url
from langchain.text_splitter import TokenTextSplitter
from DB.pg import db_conn
from LLM import embeddings_model
from DB.chroma import chroma_client
from base_model.custom_loaders import JsonObjectsLoader
import zlib


class DataIngestor:
    
    def __init__(self, user_uuid: str):
        self.chroma_client  = chroma_client.client
        self.user_uuid = user_uuid
        self.collection_name=f"{self.user_uuid}_vectorstore"
        self.chroma_vectorstore = Chroma(collection_name=self.collection_name, client=self.chroma_client, embedding_function=embeddings_model)


    def generate_content_hash(self, content) -> str:
        if isinstance(content, str):
            content_bytes = content.encode('utf-8')
        elif isinstance(content, (bytes, bytearray)):
            content_bytes = content
        else:
            raise ValueError("Content must be a string or bytes-like object")
        return zlib.crc32(content_bytes)


    def check_content_existence(self, content_hash: str):
        try:
            response =Chroma.get(self.chroma_vectorstore, where={"content_hash": content_hash }, limit=1)
            return len(response["results"]) > 0
        except Exception as e:
            raise e


    def _save_vectorstore(self, raw_documents, source, namespace):
        docs = []
        try: 
            for doc in raw_documents:
                metadata = doc.get('metadata', {})
                metadata['user_uuid'] = self.user_uuid
                metadata['source'] = source
                if namespace:
                    metadata['namespace'] = namespace
                doc  = Document(page_content=doc['page_content'], metadata=metadata)
                docs.append(doc)
            
            text_splitter = TokenTextSplitter(
                model_name="gpt-3.5-turbo",
                chunk_size=5000,
                chunk_overlap=1000,
            )
            documents = text_splitter.split_documents(docs)

            Chroma.from_documents(documents, embeddings_model, client=self.chroma_client, collection_name=self.collection_name)
            print("Vectorstore saved successfully.")
        except Exception as e:
            print(f"Failed to save vectorstore: {e}")


    def ingest(self, json_object:dict, source: str, namespace: str=None):
        raw_documents = JsonObjectsLoader()._parse(json_object)

        if namespace is not None:
            namespace = f"{self.collection_name}_{namespace}"

        content_hash = self.generate_content_hash(raw_documents)
        for doc in raw_documents:
            doc['metadata']['content_hash'] = content_hash
        
        if self.check_content_existence(content_hash):
            print("Content already exists in the vectorstore.")
            return 

        self._save_vectorstore(raw_documents, source, namespace)


    def ingest_document(self, document_path: str, document_type: str):
        loader_map = {
            'json': JSONLoader,
            'txt': UnstructuredFileLoader,
            'pdf': PyPDFLoader,
            'csv': CSVLoader,
            'xlsx': UnstructuredExcelLoader
        }

        if document_type in loader_map:
            self.ingest(loader_map[document_type], document_path, f"{document_type}_file")
        else:
            print("Unsupported document type.")
    

    def ingest_url(self, url: str):
        try:
            loader = None
            if is_youtube_url(url):
                loader = YoutubeLoader.from_youtube_url(
                    youtube_url=url,
                    add_video_info=True,
                    language=["en", "id"],
                    translation="en"
                )
            else:
                loader = WebBaseLoader(url)
                raw_documents = loader.load() if loader else []
                for doc in raw_documents:
                    if 'metadata' not in doc:
                        doc['metadata'] = {}
                    doc['metadata']['source'] = str(url)
            
            content_hash = self.generate_content_hash(raw_documents)
            if self.check_content_existence(content_hash):
                print("Content already exists in the vectorstore.")
                return 

            self._save_vectorstore(raw_documents, content_hash)
        except Exception as e:
            print(f"Error ingesting URL {url}: {e}")


    def fetch_and_ingest_sql_data(self):
        valid_tables = [
            "linkedin_features", "leet_code_features", "github_features",
            "user_profile_features", "user_file_upload", "user_url_upload"
        ]
        
        try:
            with db_conn() as conn:
                with conn.cursor() as cursor:
                    raw_documents = []
                    
                    for table_name in valid_tables: 
                        query = f"SELECT data FROM {table_name} WHERE UUID = %s"
                        cursor.execute(query, (self.user_uuid,))
                        result = cursor.fetchall()
                        
                        for row in result:
                            raw_documents.append({"page_content": row['data'], "metadata": {"source": table_name}})

                    content_hash = self.generate_content_hash(raw_documents)
                    if self.check_content_existence(content_hash):
                        print("Content already exists in the vectorstore.")
                        return 
                    self._save_vectorstore(raw_documents, content_hash)
        except Exception as e:
            print(f"Failed to fetch and ingest SQL data: {e}")