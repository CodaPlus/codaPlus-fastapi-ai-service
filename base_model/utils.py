import logging
import re
import uuid
import requests
from typing import Tuple
from typing import List
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter
from base_model.custom_wrapper import LangchainLLMWrapper
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.vectorstores.redis import Redis
from DB.redis import REDIS_VECTOR_DB
from LLM import embeddings_model


img2text_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
img2text_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


def shorten_message(message: str, max_words: int = 1000) -> str:
    is_chat_model, _, llm_model = LangchainLLMWrapper.load_llm_model(max_tokens=500)

    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=3500)
    chunks = text_splitter.split_text(message)
    docs = text_splitter.create_documents(chunks)
    tokens = [llm_model.get_num_tokens(doc.page_content) for doc in docs]
    token_count = sum(tokens)
    num_docs = len(docs)
    while token_count > 3500:
        num_docs -= 1
        token_count -= tokens[len(docs) - num_docs]
    docs = docs[:num_docs]
    shorten_message_prompt = (
        f"Given the following message, please shorten it in maximum {max_words} words.\n"
        "Message: {text}  Shorten message: "
    )
    prompt = ChatPromptTemplate.from_template(shorten_message_prompt)
    chain = load_summarize_chain(llm_model, chain_type="stuff", prompt=prompt)
    result = chain.run(docs)
    return result


def preprocess_chat_history(
    chat_history: List[dict], max_words_each_message: int = 500, max_recent_chat_history: int = 4
) -> Tuple[List[Tuple[str, str]], str, str]:
    
    image_pattern = r"---image---(.*?)---image---"
    for index, content in enumerate(chat_history):
        output_text = re.sub(image_pattern, lambda match: convert_image_to_text(match.group(1)), content["content"])
        content["content"] = output_text

    processed_chat_history = []
    question, previous_response = "", ""
    
    # Truncate to the most recent chat history if necessary
    if len(chat_history) > max_recent_chat_history:
        chat_history = chat_history[-max_recent_chat_history:]
    
    for content in chat_history:
        # Apply cleaning to the message content
        cleaned_content = remove_redundant_characters(content["content"])
        
        # Shorten the message if necessary
        content_word_count = len(cleaned_content.split())
        if content_word_count > max_words_each_message:
            cleaned_content = shorten_message(cleaned_content, max_words_each_message)
        
        # Update the content with the cleaned, potentially shortened version
        content["content"] = cleaned_content
    
    for i, item in enumerate(chat_history):
        role = item["role"]
        content = item["content"]
        
        # Combine consecutive messages from the same role
        if i > 0 and chat_history[i-1]["role"] == role:
            processed_chat_history[-1] = (processed_chat_history[-1][0], processed_chat_history[-1][1] + "\n" + content)
        else:
            processed_chat_history.append((role, content))
    
    # Extract question and previous response if available
    if processed_chat_history:
        if processed_chat_history[-1][0] == "user":
            question = processed_chat_history[-1][1]
            processed_chat_history = processed_chat_history[:-1]
        elif len(processed_chat_history) > 1:
            previous_response = processed_chat_history[-1][1]
            question = processed_chat_history[-2][1]
            processed_chat_history = processed_chat_history[:-2]
    
    # Reformat processed_chat_history to fit expected return structure
    reformatted_chat_history = []
    for item in processed_chat_history:
        if item[0] == "user":
            next_index = processed_chat_history.index(item) + 1
            if next_index < len(processed_chat_history):
                reformatted_chat_history.append((item[1], processed_chat_history[next_index][1]))
    
    return reformatted_chat_history, question, previous_response


def check_hello(message):
    lower_message = message.lower()
    question_words = [
        "who",
        "what",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "can",
        "could",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "may",
        "might",
    ]
    if "hello" in lower_message.split() or "hi" in lower_message.split():
        if not lower_message.endswith("?"):
            if not any(lower_message.startswith(word) for word in question_words):
                return True

    return False


def check_need_to_remind(message):
    if message == "":
        return False
    lower_message = message.lower()

    remind_words = [
        "sorry",
        "watch out",
        "?",
        "issue",
        "provided",
        "provide",
        "offer",
        "shipping",
        "address",
        "can",
        "replacement",
        "confirm",
    ]
    count = 0
    for word in remind_words:
        if word in lower_message:
            count += 1
            if count > 3:
                return True

    return False

def is_public_url(url):
    if 'http' not in url:
        return False
    pattern = ['docs.google.com', 'drive.google.com', 'dropbox.com']
    for p in pattern:
        if p in url:
            return False
    return True


def convert_image_to_text(image_url: str):
    try:
        raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
        inputs = img2text_processor(raw_image, return_tensors="pt")
        out = img2text_model.generate(**inputs)
        return f"---\nPicture: {img2text_processor.decode(out[0], skip_special_tokens=True)}\n---"
    except:
        return ""


def remove_redundant_characters(input_string: str) -> str:
    square_brackets_pattern = re.compile(r"\[.*?\]")
    cleaned_string = square_brackets_pattern.sub("", input_string)

    html_tags_pattern = re.compile(r"http\S+")
    cleaned_string = html_tags_pattern.sub("", cleaned_string)

    non_ascii_pattern = re.compile(r"[^\x00-\x7F]+")
    cleaned_string = non_ascii_pattern.sub("", cleaned_string)

    return cleaned_string


def in_memory_vector(conversations) -> VectorStoreRetriever:
    
    if not conversations:
        logging.error("Conversations provided.")
        return {}
    
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo", chunk_size=4000)
    chunks = text_splitter.split_text(conversations)
    docs = text_splitter.create_documents(chunks)
    
    namespace = f"IN_MEMORY_INDEX_{str(uuid.uuid4())}"
    temp_vector_store = Redis.from_documents(
        documents=docs,
        embedding=embeddings_model(),
        redis_url=REDIS_VECTOR_DB,
        index_name=namespace,
    )
    
    return temp_vector_store