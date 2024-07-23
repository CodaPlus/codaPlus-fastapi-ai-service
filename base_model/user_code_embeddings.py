from transformers import AutoTokenizer, AutoModel
import torch
from datetime import datetime
from DB.chroma import chroma_client
from base_model.custom_loaders import CodeScriptLoader
from DB.pg import db_conn


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")


class UserCodeEmbeddings:

    def __init__(self, UUID: str):
        if not UUID:
            raise ValueError("UUID is required")
        self.UUID = UUID

    def code_embeddings(self, code_snippets):
        """Generate embeddings for a list of code snippets using CodeBERT, handling long snippets.
        
        Args:
            code_snippets (List[str]): A list of code snippets as strings.
        
        Returns:
            torch.Tensor: A tensor of embeddings.
        """
        embeddings = []
        for snippet in code_snippets:
            max_length = 512
            snippet_parts = [snippet[i:i+max_length] for i in range(0, len(snippet), max_length)]
            
            snippet_embeddings = []
            for part in snippet_parts:
                inputs = tokenizer(part, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                part_embedding = outputs.last_hidden_state.mean(dim=1)
                snippet_embeddings.append(part_embedding)
            
            snippet_embedding = torch.mean(torch.stack(snippet_embeddings), dim=0)
            embeddings.append(snippet_embedding)
        
        return torch.stack(embeddings)


    def store_code_embeddings_in_chroma(self, code_snippets, metadatas):
        """
        Generate embeddings for code snippets and store them in Chroma.
        
        Args:
            code_snippets (List[str]): A list of code snippets.
            user_uuid (str): User UUID to specify the collection name.
        """
        embeddings_tensor = self.code_embeddings(code_snippets)
        collection = chroma_client.get_or_create_collection(f"{self.UUID}_code_embeddings_vectorstore")
        
        # Adding embeddings, documents, and metadatas to the collection
        try:
            collection.add(
                embeddings=embeddings_tensor.tolist(),
                documents=code_snippets,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"Error storing embedding in Chroma: {e}")


    def process(self, start_time: datetime, end_time:datetime):
        results = []
        with db_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id, question_title, difficulty, programming_language, attempted_answer FROM attempted_question WHERE created_by = %s AND completed_status = 'COMPLETED' AND created_at BETWEEN %s AND %s", (self.UUID, start_time, end_time))
                results = cursor.fetchall()

        metadatas = [{"user_uuid": self.UUID, "question_id": snippet[0], "question_title": snippet[1], "difficulty": snippet[2], "programming_language": snippet[3]} for snippet in results]
        code_snippets = [snippet[4] for snippet in results]
        self.store_code_embeddings_in_chroma(CodeScriptLoader()._parse(code_snippets, metadatas))
        print(f"Code Embeddings stored in Chroma for {self.UUID}")
