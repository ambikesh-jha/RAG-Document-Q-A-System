import ollama

def ask_external_llm(question):
    try:
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": question}]
        )

        if isinstance(response, dict) and 'message' in response:
            print(response)
            return response['message']['content']
        return "No valid response from the external LLM."
    except Exception as e:
        print(f"Error with LLM API: {e}")
        return "An error occurred while contacting the external LLM."
