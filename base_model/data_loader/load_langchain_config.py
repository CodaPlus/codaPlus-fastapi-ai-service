import yaml
from langchain.prompts.chat import ChatPromptTemplate, BaseChatPromptTemplate
from langchain.prompts import load_prompt


class LangChainDataLoader:
    """Langchain Data loader."""

    config: dict[str, dict[str, str]]
    prompts: dict[str, BaseChatPromptTemplate]

    def __init__(self):
        self.prompts = {}

        with open("../prompts/config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self._load_prompt()

    def _load_prompt(self):
        """Load prompt."""
        for title, infor in self.config.items():
            self.prompts[title] = load_prompt("./", infor["filepath"])

    def preprocessing_qa_prompt(
        self,
        rules: list[str],
        metadata: str,
        is_chat: bool,
        language: str,
        tone_of_ai: str,
        requester_name: str,
        previous_response: str,
        template: str,
        is_has_signature: bool = True,
        relevant_answer: str = None
    ):
        for prompt_title in ["qaPrompt", "qaRulePrompt"]:
            qa_template = self.prompts[prompt_title].template

            if relevant_answer:
                qa_template += (
                    f"This is a relevant answer that you should use information from it to generate response.\n"
                    f"Relevant answer: {relevant_answer}"
                )

            if is_chat:
                qa_template += (
                    "\nNEVER tell the user that you will check for them."
                    "\nNEVER tell the user that you are reaching out to anyone else."
                    "\nNEVER tell the user to wait for you get back to them."
                )

            else:
                if requester_name != "":
                    qa_template += f"Customer's name: {requester_name}\n\n"

                if previous_response != "":
                    qa_template += (
                        f"This is the latest response from you:\n"
                        f"'{previous_response}'\n"
                        f"The user has not replied anything from this point."
                        f"Generate a new follow up message in {language} language."
                    )  # noqa
                
                if template != "":
                    template = template.replace("{", "<").replace("}", ">")
                    qa_template += (
                        f" with the following template, please translate the template into "
                        f"{language} language as the final response in markdown format:"
                        f"\n'{template}'\n"
                    )  # noqa

                qa_template += (
                    "NEVER redirect this issue or connect or reach out to the support team, "
                    "you are working as a support user agent in this team right now.\n"
                )

            if is_has_signature:
                qa_template += (
                    'Please DO NOT add any close mark (such as "Best regards", "Sincerely", '
                    '"Take care", "Your ...", etc) and the signature in the end of your answer.'
                )
            qa_template += f"\nWHEN ANSWERING PLEASE USE THE TONE IF GIVEN AS: {tone_of_ai} IN CHAT STYLE (not email style). \nResponse:\n\n"

            qa_template = qa_template.format(
                rules="".join(f"- {x}\n" for x in rules),
                metadata=metadata,
                context="{context}",
                question="{question}",
            )
            self.prompts[prompt_title] = ChatPromptTemplate(template=qa_template, input_variables=["context", "question"])
