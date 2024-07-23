from __future__ import annotations
import json
import logging
import re
from typing import Tuple, Union, List
import yaml
from langchain.chains.llm import LLMChain
from langchain.vectorstores.chroma import Chroma
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import MergerRetriever
from langchain.schema import HumanMessage
from langchain.vectorstores.base import VectorStoreRetriever, VectorStore
from base_model.retrieval_chain import CustomConversationalRetrievalChain
from .data_loader.load_langchain_config import LangChainDataLoader
from .data_loader.vector_store_retriever import CustomVectorStoreRetriever, CustomMergerRetriever
from DB.chroma import chroma_client
from LLM import embeddings_model, DEFAULT_MODEL
from langchain.llms import BaseLLM


class LangchainLLMWrapper:

    def __init__(
        self,
        vectorstore: VectorStore,
        vectorstore_retriever: VectorStoreRetriever = None,
        response_lib_retriever: MergerRetriever = None,
        rules: list[str] = None,
        requester_name: str = "",
        tone_of_ai: str = "friendly",
        language: str = "English",
        template: str = "",
        previous_response: str = "",
        is_chat: bool = False,
        metadata: list[dict] = None,
        brand_tone: dict = None,
        question: str = "",
    ):
        self.rules = rules or []
        self.is_only_rspl = False 
        if is_chat:
            self.rules = self._add_default_rules(self.rules)

        self.is_chat_model, self.llm_cls, self.llm_model = self.load_llm_model()
        self.data_loader = LangChainDataLoader()
        with open("./prompts/supported_languages.yaml") as f:
            supported_languages = yaml.safe_load(f)
        if language.lower() not in supported_languages["support_languages"]:
            language = self._detect_language(question)

        self.most_relevant_answer = ""
        self.relevant_answer = ""
        self.score = None

        self.data_loader.preprocessing_qa_prompt(
            rules=self.rules,
            metadata=self._format_dict_list(metadata or []),
            is_chat=is_chat,
            brand_tone=brand_tone,
            language=language,
            tone_of_ai=tone_of_ai,
            requester_name=requester_name,
            previous_response=previous_response,
            template=template,
            relevant_answer = self.relevant_answer if self.relevant_answer != "" else None
        )

        self.vectorstore = vectorstore
        self.vectorstore_retriever = vectorstore_retriever
        self.response_lib_retriever = response_lib_retriever
        self.is_chat = is_chat
        
        logging.info(f"Relevant answer: {self.relevant_answer}")


    @staticmethod
    def get_single_retriever(namespace: Union[List[str], str], search_kwargs: dict = None) -> MergerRetriever:
        if search_kwargs is None:
            search_kwargs = {"k": 1, "score_threshold": 0.3}
        try:
            embeddings = embeddings_model
            response_lib_vectorstore = Chroma(embedding_function=embeddings, collection_name=namespace, client=chroma_client.client)
            retrievers = CustomVectorStoreRetriever(
                vectorstore=response_lib_vectorstore,
                search_type="similarity_score_threshold",
                search_kwargs=search_kwargs,
            )
            return CustomMergerRetriever(retrievers=[retrievers])
        except Exception as e:  
            raise Exception(f"Error when loading response lib vectorstore. {e}")


    @staticmethod
    def get_langchain_retriever(
        namespaces: Union[List[str], str],
        vts_search_kwargs: dict = None,
        th_vts_search_kwargs: dict = None,
    ) -> Tuple[VectorStore, MergerRetriever]:

        if vts_search_kwargs is None:
            vts_search_kwargs = {"k": 3, "score_threshold": 0.3}
        if th_vts_search_kwargs is None:
            th_vts_search_kwargs = {"score_threshold": 0.3, "k": 2}

        try:
            embeddings = embeddings_model
            retrievers = []
            vectorstore = None
            try:
                for namespace in namespaces:
                    vectorstore = Chroma(embedding_function=embeddings, collection_name=namespace, client=chroma_client.client)
                    try: 
                        help_center_retriever = CustomVectorStoreRetriever(
                            vectorstore=vectorstore,
                            search_type="similarity_score_threshold",
                            search_kwargs=vts_search_kwargs,
                        )
                        retrievers.append(help_center_retriever)
                    except Exception as e:
                        print(e)
            except Exception as e:
                print(e)

            vectorstore_retriever = MergerRetriever(retrievers=retrievers)

            return vectorstore, vectorstore_retriever
        except Exception as e: 
            print(e)
            raise Exception(f"Error when loading vectorstore. {e}")


    def _detect_language(self, question: str) -> str:
        try:
            language_detect_prompt = self.data_loader.prompts.get("detectLanguagePrompt").template.format(
                question=question,
                chat_history="",
            )
            detected_language = (
                self.llm_model.generate([[HumanMessage(content=language_detect_prompt)]]).generations[0][0].text.strip()
            )
            regex = r"\%([^%]+)\%"
            language = re.findall(regex, detected_language)[-1]

        except Exception as e:
            language = "English"
        return language


    @staticmethod
    def load_llm_model(max_tokens: int = None, llm_model = DEFAULT_MODEL):

        if max_tokens is not None:
            llm_model.max_tokens = max_tokens
        return True, BaseLLM, llm_model


    def get_chain(self) -> ConversationalRetrievalChain:
        prompt_title = "qaRulePrompt" if self.rules else "qaPrompt"

        docs_chain = load_qa_chain(self.llm_model, prompt=self.data_loader.prompts[prompt_title])
        return CustomConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever()
            if self.vectorstore_retriever is None
            else self.vectorstore_retriever,
            combine_docs_chain=docs_chain,
            question_generator=LLMChain(llm=self.llm_model, prompt=self.data_loader.prompts["condensePrompt"]),
            max_tokens_limit=2500,
            is_chat=self.is_chat,
            most_relevant_answer = self.most_relevant_answer,
            best_answer_score = self.score
        )


    def get_stream_chain(self, stream_handler) -> ConversationalRetrievalChain:
        manager = AsyncCallbackManager([])
        stream_manager = AsyncCallbackManager([stream_handler])

        prompt_title = "qaRulePrompt" if self.is_chat or self.rules else "qaPrompt"
        llm = self.llm_cls(temperature=0, streaming=True, callback_manager=stream_manager)
        docs_chain = load_qa_chain(llm, prompt=self.data_loader.prompts[prompt_title])

        return CustomConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever()
            if self.vectorstore_retriever is None
            else self.vectorstore_retriever,
            combine_docs_chain=docs_chain,
            question_generator=LLMChain(llm=self.llm_model, prompt=self.data_loader.prompts["condensePrompt"]),
            callback_manager=manager,
            max_tokens_limit=2500,
            is_chat=self.is_chat,
            most_relevant_answer = self.most_relevant_answer,
            best_answer_score = self.score
        )


    @staticmethod
    def _add_default_rules(rules: list[str]):
        rules.extend(
            [
                "Do not say I'm sorry when user raise any issue.",
                "Do not say the user to contact you at the end of the response if they don't ask for the contact.",
                "If the user do not greet you, do not greet them back.",
                "Dont mention your name in the response.",
            ]
        )
        return rules


    @staticmethod
    def _add_remove_signature_rule(rules: list[str]):
        rules.append(
            'Please DO NOT add any close mark (such as "Best regards", "Sincerely", "Take care", "Your ...", etc) '
            "and the signature in the end of your answer."
        )
        return rules


    def _format_dict_list(self, dict_list: list[dict]):
        result = ""
        for item in dict_list:
            for category, info in item.items():
                result += f"{category.capitalize().replace('_', ' ')}: \n"
                result += json.dumps(info, indent=4).replace("{", "<").replace("}", ">")
                result += "\n\n"
        return result




    ## CHAT RELATED METHODS ##

    def get_chat_history_summary(self, chat_history: list) -> str:
        if not len(chat_history):
            return ""

        chat_history_str = self._get_chat_history_str(chat_history)

        recap_chat_history_prompt = self.data_loader.prompts["recapChatHistoryPrompt"].format(
            chat_history=chat_history_str,
        )
        if self.is_chat_model:
            chat_history_summary = self.llm_model.generate([[HumanMessage(content=recap_chat_history_prompt)]])
        else:
            chat_history_summary = self.llm_model.generate([recap_chat_history_prompt])

        if chat_history_summary:
            chat_history_summary = chat_history_summary.generations[0][0].text.strip()

        return chat_history_summary
    

    def _get_chat_history_str(self, chat_history: list) -> str:
        buffer = ""
        for dialogue_turn in chat_history:
            user = f"User: {dialogue_turn['User']}"
            ai = f"Agent: {dialogue_turn['Agent']}"
            buffer += "\n" + "\n".join([user, ai])
        return buffer


    def get_relevant_documents(self, question: str, chat_history: list, top_k: int = 4) -> list:
        """Get document sources for question/answering."""
        chat_history_str = self._get_chat_history_str(chat_history)

        condense_question_prompt = self.data_loader.prompts["condensePrompt"].format(
            question=question,
            chat_history=chat_history_str,
        )
        if self.is_chat_model:
            condense_question = self.llm_model.generate([[HumanMessage(content=condense_question_prompt)]])
        else:
            condense_question = self.llm_model.generate([condense_question_prompt])

        if condense_question:
            condense_question = condense_question.generations[0][0].text.strip()

        search_result = self.vectorstore.similarity_search_with_relevance_scores(condense_question)

        return search_result[:top_k]