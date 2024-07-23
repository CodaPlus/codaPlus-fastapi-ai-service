from langchain_openai import ChatOpenAI
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os 
from dotenv import load_dotenv
load_dotenv()


DEFAULT_MODEL = ChatOpenAI(
    api_key=str(os.getenv("LlAMA_CPP_API_KEY")),
    base_url=str(os.getenv("LlAMA_CPP_Feature_Model_BASE_URL")),
    model_name="fintuned-feature-generation",
    temperature=0.4
)


FINETUNED_MODEL = ChatOpenAI(
    api_key=str(os.getenv("LlAMA_CPP_API_KEY")),
    base_url=str(os.getenv("LlAMA_CPP_QandA_Model_BASE_URL")),
    model_name="fintuned-qanda-generation",
    temperature=0.4
)


## Embeddings
embeddings_model = SentenceTransformerEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1",
    model_kwargs = {'device': 'cpu'},
    cache_folder='./ml_models'
)


code_embeddings = SentenceTransformerEmbeddings(
    model_name="microsoft/codebert-base",
    model_kwargs = {'device': 'cpu'},
    cache_folder='./ml_models'
)