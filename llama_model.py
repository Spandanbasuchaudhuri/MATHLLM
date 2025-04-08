import ollama

def query_llama3_stream(prompt: str):
    """
    Streams output from LLaMA 3 using the ollama Python package.
    Yields text chunks.
    """
    stream = ollama.chat(
        model='llama3:8b',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True
    )
    
    for chunk in stream:
        if 'message' in chunk and 'content' in chunk['message']:
            yield chunk['message']['content']