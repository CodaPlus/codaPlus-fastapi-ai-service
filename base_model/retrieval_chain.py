import inspect
from typing import Dict, Any, Optional, List, Tuple
from langchain.callbacks.manager import CallbackManagerForChainRun, AsyncCallbackManagerForChainRun
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain, _get_chat_history
from langchain.load.dump import dumpd
from langchain.retrievers import MergerRetriever
from langchain.schema import BaseOutputParser, Document

import spacy
nlp = spacy.load("xx_ent_wiki_sm")

class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    retriever: MergerRetriever
    output_parser: BaseOutputParser = None
    is_chat: bool = False
    most_relevant_answer: str = None
    best_answer_score: float = None

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])

        if self.most_relevant_answer != "" and self.best_answer_score >= 0.9:
            output: Dict[str, Any] = {"answer": self.most_relevant_answer, "score": self.best_answer_score}
            if self.output_parser:
                output.update({"answer": self.output_parser.parse(output["answer"])})
            return output

        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = self.question_generator.run(
                question=question, chat_history=chat_history_str, callbacks=callbacks
            )
        else:
            new_question = question

        accepts_run_manager = "run_manager" in inspect.signature(self._get_docs).parameters
        if accepts_run_manager:
            docs, scores = self._get_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs, scores = self._get_docs(new_question, inputs)  # type: ignore[call-arg]

        new_inputs = inputs.copy()
        if self.rephrase_question:
            new_inputs["question"] = new_question

        new_inputs["chat_history"] = chat_history_str

        answer = self.combine_docs_chain.run(input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs)

        output: Dict[str, Any] = {self.output_key: answer, "scores": scores}
        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_question
        if self.output_parser is not None:
            output.update({"answer": self.output_parser.parse(output["answer"])})
        return output

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if self.most_relevant_answer != "" and self.best_answer_score >= 0.9:
            output: Dict[str, Any] = {"answer": self.most_relevant_answer, "score": self.best_answer_score}
            if self.output_parser:
                output.update({"answer": self.output_parser.parse(output["answer"])})
            return output

        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str, callbacks=callbacks
            )
        else:
            new_question = question

       
        accepts_run_manager = "run_manager" in inspect.signature(self._aget_docs).parameters
        if accepts_run_manager:
            docs, scores = await self._aget_docs(new_question, inputs, run_manager=_run_manager)
        else:
            docs, scores = await self._aget_docs(new_question, inputs)  # type: ignore[call-arg]

        new_inputs = inputs.copy()
        if self.rephrase_question:
            new_inputs["question"] = new_question
        new_inputs["chat_history"] = chat_history_str
        answer = await self.combine_docs_chain.arun(
            input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
        )
        output: Dict[str, Any] = {self.output_key: answer, "scores": scores}
        if self.return_source_documents:
            output["source_documents"] = docs
        if self.return_generated_question:
            output["generated_question"] = new_question

        if self.output_parser is not None:
            output.update({"answer": await self.output_parser.aparse(output["answer"])})

        return output

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[list[Document], list[float]]:
        """Get docs."""

        from langchain.callbacks.manager import CallbackManager

        callback_manager = CallbackManager.configure(
            run_manager.get_child(),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = callback_manager.on_retriever_start(
            dumpd(self),
            question,
            **kwargs,
        )
        try:
            # Get the results of all retrievers.
            retriever_docs = [
                (
                    retriever.get_relevant_documents(
                        question, callbacks=run_manager.get_child("retriever_{}".format(i + 1))
                    ),
                    retriever.metadata["name"],
                )
                for i, retriever in enumerate(self.retriever.retrievers)
            ]

            # Merge the results of the retrievers.
            merged_documents = []
            merged_scores = []

            max_docs = max(len(docs[0]) for docs in retriever_docs) if len(retriever_docs) > 0 else 0
            for i in range(max_docs):
                for retriever, doc_with_score in zip(self.retriever.retrievers, retriever_docs):
                    if i < len(doc_with_score[0]):
                        merged_documents.append(doc_with_score[0][i][0])
                        merged_scores.append({"tag": doc_with_score[1], "score": doc_with_score[0][i][1]})

        except Exception as e:
            run_manager.on_retriever_error(e)
            raise e
        else:
            run_manager.on_retriever_end(
                merged_documents,
                **kwargs,
            )

        docs, scores = self._reduce_docs(merged_documents, merged_scores, **kwargs)
        return docs, scores

    @staticmethod
    def find_persons(text):
        doc2 = nlp(text)
        persons = [ent.text for ent in doc2.ents if ent.label_ == "PER"]
        return persons
    
    @staticmethod
    def find_orgs(text):
        doc2 = nlp(text)
        orgs = [ent.text for ent in doc2.ents if ent.label_ == "ORG"]
        return orgs

    def _reduce_docs(self, merged_documents, merged_scores, **kwargs):
        docs, scores = self._reduce_tokens_below_limit_with_score(merged_documents, merged_scores)
        if self.is_chat:
            new_docs, new_scores = [], []
            for idx, doc in enumerate(docs):
                if scores[idx]["tag"] == "chatbot":
                    if len(self.find_persons(doc.page_content)) < 2:
                        new_docs.append(doc)
                        new_scores.append(scores[idx])
                else:
                    new_docs.append(doc)
                    new_scores.append(scores[idx])
            docs, scores = new_docs, new_scores
        return docs, scores

    def _reduce_tokens_below_limit_with_score(
        self, docs: List[Document], scores: List = None
    ) -> Tuple[List[Document], List[float]]:
        num_docs = len(docs)

        if self.max_tokens_limit and isinstance(self.combine_docs_chain, StuffDocumentsChain):
            tokens = [self.combine_docs_chain.llm_chain.llm.get_num_tokens(doc.page_content) for doc in docs]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs], scores[:num_docs]

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> tuple[list[Document], list[float]]:
        """Get docs."""

        from langchain.callbacks.manager import AsyncCallbackManager

        callback_manager = AsyncCallbackManager.configure(
            run_manager.get_child(),
            None,
            verbose=kwargs.get("verbose", False),
            inheritable_tags=tags,
            local_tags=self.tags,
            inheritable_metadata=metadata,
            local_metadata=self.metadata,
        )
        run_manager = await callback_manager.on_retriever_start(
            dumpd(self),
            question,
            **kwargs,
        )
        try:
            # Get the results of all retrievers.
            retriever_docs = [
                (
                    await retriever.aget_relevant_documents(
                        question, callbacks=run_manager.get_child("retriever_{}".format(i + 1))
                    ),
                    retriever.metadata["name"],
                )
                for i, retriever in enumerate(self.retriever.retrievers)
            ]

            # Merge the results of the retrievers.
            merged_documents = []
            merged_scores = []
            max_docs = max(len(docs[0]) for docs in retriever_docs) if len(retriever_docs) > 0 else 0
            for i in range(max_docs):
                for retriever, doc_with_score in zip(self.retriever.retrievers, retriever_docs):
                    if i < len(doc_with_score[0]):
                        merged_documents.append(doc_with_score[0][i][0])
                        merged_scores.append({"tag": doc_with_score[1], "score": doc_with_score[0][i][1]})

        except Exception as e:
            await run_manager.on_retriever_error(e)
            raise e
        else:
            await run_manager.on_retriever_end(
                merged_documents,
                **kwargs,
            )

        docs, scores = self._reduce_docs(merged_documents, merged_scores, **kwargs)
        return docs, scores
