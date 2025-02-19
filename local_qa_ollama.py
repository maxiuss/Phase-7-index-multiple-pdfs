# local_qa_ollama.py
import os
from dotenv import load_dotenv
from semantic_search.semantic_search import semantic_search  # Your existing semantic search code
from ollama_client import query_ollama

load_dotenv()  # Load any necessary configuration

def extract_text(chunk):
    """
    Extract the text from a chunk.
    If 'chunk' is a dict and its 'text' value is also a dict, try extracting the inner string.
    """
    if isinstance(chunk, dict) and "text" in chunk:
        text_val = chunk["text"]
        if isinstance(text_val, dict) and "text" in text_val:
            return text_val["text"]
        elif isinstance(text_val, str):
            return text_val
        else:
            return str(text_val)
    return str(chunk)

def answer_question_with_ollama(question, k=3, model="llama2"):
    """
    Uses your FAISS index to retrieve context and then queries Ollama for an answer.
    """
    # 1. Retrieve relevant text chunks.
    context_chunks = semantic_search(question, k=k)
    # Extract the text using our helper function.
    # Limit each chunk to 500 characters
    context = "\n".join(extract_text(chunk)[:500] for chunk in context_chunks[:3])

    # 2. Build a prompt that includes the retrieved context and the question.
    prompt = (
        f"Using the following context:\n{context}\n\n"
        f"Answer the question: {question}\nAnswer:"
    )

    # 3. Query Ollama with the prompt.
    answer = query_ollama(prompt, model=model)
    return answer

if __name__ == "__main__":
    # Test the integration
    question = input("Enter your question: ")
    answer = answer_question_with_ollama(question)
    print("\nAnswer from Ollama:")
    print(answer)

