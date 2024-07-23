from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
import numpy as np
import json
import pandas as pd
import tiktoken
from langchain.schema import HumanMessage, SystemMessage
from base_model.custom_wrapper import LangchainLLMWrapper

ZERO_SHOT_CLF_PROMPT_TEMPLATE = """
You will be provided with the following information:
1. An arbitrary text sample. The sample is delimited with triple backticks.
2. List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.

Perform the following tasks:
1. Identify to which category the provided text belongs to with the highest probability.
2. Assign the provided text to that category.
3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.

List of categories: {labels}

Text sample: ```{x}```

Your JSON response:
"""



def to_numpy(X: Any) -> np.ndarray:
    """Converts a pandas Series or list to a numpy array.

    Parameters
    ----------
    X : Any
        The data to convert to a numpy array.

    Returns
    -------
    X : np.ndarray
    """
    if isinstance(X, pd.Series):
        X = X.to_numpy().astype(object)
    elif isinstance(X, list):
        X = np.asarray(X, dtype=object)
    if isinstance(X, np.ndarray) and len(X.shape) > 1:
        X = np.squeeze(X)
    return X


def find_json_in_string(string: str) -> str:
    """Finds the JSON object in a string.

    Parameters
    ----------
    string : str
        The string to search for a JSON object.

    Returns
    -------
    json_string : str
    """
    start = string.find("{")
    end = string.rfind("}")
    if start != -1 and end != -1:
        json_string = string[start : end + 1]
    else:
        json_string = "{}"
    return json_string


def extract_json_key(json_: str, key: str):
    """Extracts JSON key from a string.

    json_ : str
        The JSON string to extract the key from.
    key : str
        The key to extract.
    """
    original_json = json_
    for i in range(2):
        try:
            json_ = original_json.replace("\n", "")
            if i == 1:
                json_ = json_.replace("'", '"')
            json_ = find_json_in_string(json_)
            as_json = json.loads(json_)
            if key not in as_json.keys():
                raise KeyError("The required key was not found")
            return as_json[key]
        except Exception:
            if i == 0:
                continue
            return None

class BaseClassifier(ABC):
    default_label: Optional[str] = "Undefined"

    def _to_np(self, X):
        """Converts X to a numpy array.

        Parameters
        ----------
        X : Any
            The input data to convert to a numpy array.

        Returns
        -------
        np.ndarray
            The input data as a numpy array.
        """
        return to_numpy(X)

    @abstractmethod
    def _predict_single(self, x: str) -> Any:
        """Predicts the class of a single input."""
        pass

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.Series, List[str]]],
        y: Union[np.ndarray, pd.Series, List[str], List[List[str]]],
    ):
        """Extracts the target for each datapoint in X.

        Parameters
        ----------
        X : Optional[Union[np.ndarray, pd.Series, List[str]]]
            The input array data to fit the model to.

        y : Union[np.ndarray, pd.Series, List[str], List[List[str]]]
            The target array data to fit the model to.
        """
        X = self._to_np(X)
        self.classes_ = y
        return self

    def predict(self, X: Union[np.ndarray, pd.Series, List[str]]):
        """Predicts the class of each input.

        Parameters
        ----------
        X : Union[np.ndarray, pd.Series, List[str]]
            The input data to predict the class of.

        Returns
        -------
        List[str]
        """
        X = self._to_np(X)
        predictions = []
        for i in range(len(X)):
            predictions.append(self._predict_single(X[i]))
        return predictions

    def _extract_labels(self, y: Any) -> List[str]:
        """Return the class labels as a list.

        Parameters
        ----------
        y : Any

        Returns
        -------
        List[str]
        """
        if isinstance(y, (pd.Series, np.ndarray)):
            labels = y.tolist()
        else:
            labels = y
        return labels

    def _get_default_label(self):
        """Returns the default label based on the default_label argument."""
        return self.default_label


class ZeroShotGPTClassifier(BaseClassifier):
    """Base class for zero-shot classifiers.

    Parameters
    ----------
    openai_key : Optional[str] , default : None
    openai_org : Optional[str] , default : None
    gpt_model : str , default : "gpt-3.5-turbo"
    default_label : Optional[Union[List[str], str]] , default : 'Random'
        The default label to use if the LLM could not generate a response for a sample. If set to 'Random' a random
        label will be chosen based on probabilities from the training set.
    prompt_template: str , A formattable string with the following placeholders: {x} - the sample to classify, {labels} - the list of labels.
        If None, the default prompt template will be used.
    """
    MAX_TOKENS_RESPONSE = 60
    MAX_TOKENS_MESSAGE = 3826

    def __init__(
        self,
        default_label: Optional[Union[List[str], str]] = "Random",
        prompt_template: Optional[str] = None,
    ):
        self.is_chat_model, _, self.llm_model = LangchainLLMWrapper.load_llm_model(ZeroShotGPTClassifier.MAX_TOKENS_RESPONSE)
        self.llm_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.default_label = default_label
        self.prompt_template = prompt_template

    def _get_prompt(self, x) -> str:
        if self.prompt_template is None:
            return ZERO_SHOT_CLF_PROMPT_TEMPLATE.format(x = x, labels = repr(self.classes_))
        return self.prompt_template.format(
            x=x, labels = repr(self.classes_)
        )

    def _get_chat_completion(self, x):
        prompt = self._get_prompt(x)
        msgs = []
        msgs.append(SystemMessage(content="You are a text classification model."))
        msgs.append(HumanMessage(content=prompt))
        completion = self.llm_model.generate([msgs])
        return completion

    def _predict_single(self, x):
        """Predicts the labels for a single sample.

        Should work for all (single label) GPT based classifiers.
        """
        encoded_x = self.llm_encoding.encode(x)
        current_length = len(encoded_x)
        if current_length > ZeroShotGPTClassifier.MAX_TOKENS_MESSAGE:
            x = encoded_x[:ZeroShotGPTClassifier.MAX_TOKENS_MESSAGE // 2] + encoded_x[-ZeroShotGPTClassifier.MAX_TOKENS_MESSAGE // 2:]
            x = self.llm_encoding.decode(x)
        completion = self._get_chat_completion(x)
        try:
            label = str(
                extract_json_key(
                    completion.generations[0][0].text.strip(), "label"
                )
            )
        except Exception as e:
            print("Could not extract the label from the completion", e)
            label = ""

        if label not in self.classes_:
            label = label.replace("'", "").replace('"', "")
            if label not in self.classes_:  # try again
                label = self._get_default_label()
        return label
        