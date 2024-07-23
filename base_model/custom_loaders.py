import json
import base64
from typing import Any, Dict, Iterator
from langchain.schema import Document


class JsonObjectsLoader:

    def _parse(self, content: Dict[str, Any]) -> Iterator[Document]:
        """
        Convert given JSON content to documents containing only text.
        
        :param content: JSON object containing the content.
        :return: Iterator of Document instances with only page_content.
        """
        
        data = content if isinstance(content, list) else [content]

        for sample in data:
            text = self._get_text(sample=sample)
            print(f"Text: {text}")
            yield Document(page_content=text, metadata={})


    def _get_text(self, sample: Any) -> str:
        """
        Convert sample to string format. Simplified to directly handle JSON objects.
        
        :param sample: A single data payload, which could be a string, dictionary, or other types.
        :return: Text extracted or converted from the sample.
        """
        print(f"Sample: {sample}")
        if isinstance(sample, str):
            return sample
        elif isinstance(sample, dict):
            return json.dumps(sample) if sample else ""
        else:
            return str(sample) if sample is not None else ""


class CodeScriptLoader:

    def _parse(self, content: Dict[str, Any]) -> Iterator[Document]:
        """
        Convert given base64-encoded content to documents containing only code scripts.
        
        :param content: JSON object containing the base64-encoded content.
        :return: Iterator of Document instances with only page_content as the decoded code script.
        """
        
        data = content if isinstance(content, list) else [content]
        for sample in data:
            code_script = self._get_text(sample=sample)
            print(f"Code Script: {code_script}")
            yield Document(page_content=code_script, metadata={})

    def _get_text(self, sample: Any) -> str:
        """
        Decode sample from base64 format to a string format. Simplified to directly handle JSON objects.
        
        :param sample: A single data payload, which could be a base64-encoded string, dictionary, or other types.
        :return: Text (code script) decoded from the sample.
        """
        print(f"Sample: {sample}")
        if isinstance(sample, str):
            return base64.b64decode(sample).decode('utf-8')
        elif isinstance(sample, dict):
            return json.dumps(sample) if sample else ""
        else:
            return str(sample) if sample is not None else ""
