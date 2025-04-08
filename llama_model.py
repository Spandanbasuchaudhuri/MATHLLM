import ollama

def query_llama3_stream(prompt: str):
    """
    Streams output from LLaMA 3 using the ollama Python package.
    Yields text chunks.
    """
    # Create a streaming response from ollama
    stream = ollama.chat(
        model='llama3:8b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    
    # Yield each chunk as it arrives
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']