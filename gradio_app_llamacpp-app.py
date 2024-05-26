import gradio as gr
from llama_cpp import Llama


def llama_cpp_chat(gguf_model, prompt:str, messages:str = ''):
    prompt_templated = f'{messages}\n ### HUMAN:\n{prompt} \n ### ASSISTANT:'
    output = gguf_model(
          prompt_templated, # Prompt
          max_tokens=512, 
          stop=["### HUMAN:\n", " ### ASSISTANT:"], # Stop generating just before the model would generate a new question
          echo=True # Echo the prompt back in the output
    ) # Generate a completion, can also call create_completion
    print(output)
    return output['choices'][0]['text']

llm = Llama(
      model_path="llama3_8b_chat_brainstorm.Q2_K.gguf",
      # n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

def chatty(prompt, messages):
    print(prompt)
    print(f'messages: {messages}')
    past_messages = ''
    if len(messages) > 0:
        for idx, message in enumerate(messages):
            print(f'idx: {idx}, message: {message}')
            past_messages += f'\n### HUMAN: {message[0]}'
            past_messages += f'\n### ASSISTANT: {message[1]}'
                
                
        # past_messages = messages[0][0]
    print(f'past_messages: {past_messages}')
    messages = llama_cpp_chat(llm, prompt, past_messages)
    return messages.split('### ASSISTANT:')[-1]


demo = gr.ChatInterface(
    fn=chatty,
    title="Brainstorm on CPU with llama.cpp",
    description="Please note that CPU prediction will very slow - but this can run on the Free Tier :)"
) 


if __name__ == "__main__":
    demo.launch()