import asyncio
from functools import partial
from typing import List, Any, Dict, ClassVar, Collection
from pydantic import Field
from langchain.callbacks.manager import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain.schema import Document, BaseRetriever
from langchain.retrievers import MergerRetriever
from langchain.vectorstores.chroma import Chroma


class CustomVectorStoreRetriever(BaseRetriever):
    """Retriever class for VectorStore."""

    vectorstore: Chroma
    """VectorStore to use for retrieval."""
    search_type: str = "similarity_search"
    """Type of search to perform. Defaults to "similarity_search"."""
    search_kwargs: dict = Field(default_factory=dict)
    """Keyword arguments to pass to the search function."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity_search", #similarity_search
        "similarity_score_threshold",
        "mmr",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List:
        if self.search_type == "similarity_search":
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
        elif self.search_type == "similarity_score_threshold":
            docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs)

        elif self.search_type == "mmr":
            embedding = self.vectorstore._embedding_function(query)
            docs_with_scores = self.vectorstore.max_marginal_relevance_search_by_vector( ## max_marginal_relevance_search_with_score_by_vector
                embedding, **self.search_kwargs
            )
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")

        return docs_with_scores

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List:
        if self.search_type == "similarity_search":
            func = partial(self.vectorstore.similarity_search_with_score, query, **self.search_kwargs)
            docs_with_scores = await asyncio.get_event_loop().run_in_executor(None, func)
        elif self.search_type == "similarity_score_threshold":
            func = partial(
                self.vectorstore.similarity_search_with_relevance_scores,
                query,
                **self.search_kwargs,
            )

            docs_with_scores = await asyncio.get_event_loop().run_in_executor(None, func)
        elif self.search_type == "mmr":
            embedding = self.vectorstore.embedding_function(query)
            func = partial(
                self.vectorstore.as_retriever(search_type="mmr"),
                embedding,
                **self.search_kwargs,
            )
            docs_with_scores = await asyncio.get_event_loop().run_in_executor(None, func)
        else:
            raise ValueError(f"search_type of {self.search_type} not allowed.")
        return docs_with_scores

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return await self.vectorstore.aadd_documents(documents, **kwargs)

class CustomMergerRetriever(MergerRetriever):
    async def merge_documents_and_scores(self, query: str, run_manager: AsyncCallbackManagerForRetrieverRun) -> List:
        """
        Merge the results of the retrievers.

        Args:
            query: The query to search for.

        Returns:
            A list of merged documents.
        """

        # Get the results of all retrievers.
        retriever_docs = [
            (
                await retriever.aget_relevant_documents(
                    query, callbacks=run_manager.get_child("retriever_{}".format(i + 1))
                ),
                retriever.metadata["name"],
            )
            for i, retriever in enumerate(self.retrievers)
        ]

        # Merge the results of the retrievers.
        merged_documents = []
        merged_scores = []

        max_docs = max(len(docs[0]) for docs in retriever_docs) if len(retriever_docs) > 0 else 0
        for i in range(max_docs):
            for doc_with_score in retriever_docs:
                if i < len(doc_with_score[0]):
                    merged_documents.append(doc_with_score[0][i][0])
                    merged_scores.append(doc_with_score[0][i][1])
        
        if not merged_documents or not merged_scores:
            return [], []
        else:
            merged_documents, merged_scores = zip(
                *sorted(zip(merged_documents, merged_scores), key=lambda x: x[1], reverse=True)
            )

        return merged_documents, merged_scores