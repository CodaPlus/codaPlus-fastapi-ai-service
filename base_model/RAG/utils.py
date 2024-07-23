from unstructured.partition.xlsx import partition_xlsx
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from typing import Any, List
from urllib.parse import urlparse


class UnstructuredExcelLoader(UnstructuredFileLoader):
    def __init__(self, file_path: str, mode: str = "single", **unstructured_kwargs: Any):
        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        return partition_xlsx(filename=self.file_path, **self.unstructured_kwargs)


def is_youtube_url(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    return "youtube.com" in domain or "youtu.be" in domain