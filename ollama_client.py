import subprocess

def query_ollama(prompt, model="llama2"):
    """
    Query a locally running model via Ollama.

    :param prompt: The prompt string to send.
    :param model: The model name to run (default: "llama2").
    :return: The generated text (string).
    """
    # Remove the flag and pass the prompt via standard input
    command = ["ollama", "run", model]
    try:
        result = subprocess.run(command, input=prompt, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Ollama query failed: {e.stderr}")

# Example usage:
if __name__ == "__main__":
    sample_prompt = "Explain the process of photosynthesis."
    response = query_ollama(sample_prompt)
    print("Response from Ollama:")
    print(response)

