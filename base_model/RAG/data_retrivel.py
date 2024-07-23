from typing import Any, Optional, Union
from DB.chroma import chroma_client
from LLM import embeddings_model
from langchain.vectorstores.chroma import Chroma


class DataRetriever:
    def __init__(self, user_uuid: str):
        self.chroma_client = chroma_client.client
        self.user_uuid = user_uuid
        self.chroma_vectorstore = Chroma(collection_name=f"{self.user_uuid}_vectorstore", client=self.chroma_client, embedding_function=embeddings_model)


    def retriever(self, query: str, k=4, query_filter: Optional[dict] = None) -> Union[list, None]:
        """Retrieve documents based on similarity to the query without scores."""
        retrieved_documents = Chroma.similarity_search(self.chroma_vectorstore, query=query, k=k, filter=query_filter)
        return retrieved_documents


    def retriever_similarity_search_with_score(self, query: str, k=4, query_filter: Optional[dict] = None) -> Union[list, None]:
        """Retrieve documents based on similarity to the query without scores."""
        retrieved_documents = Chroma.similarity_search_with_score(embeddings_model, client=self.chroma_client, query=query, k=k, filter=query_filter, collection_name=self.collection_name)
        return retrieved_documents