import gradio as gr
import os
import requests
import json

sbc_host_url = os.environ['URL']

# def get_completion(prompt:str, messages:str = '', n_predict=128):
#     system = "### System: You are a helpful assistant helps to brainstorm ideas.\n"
#     prompt_templated = f'{system} {messages}\n ### HUMAN:\n{prompt} \n ### ASSISTANT:'

#     headers = {
#         "Content-Type": "application/json"
#     }
#     data = {
#         "prompt": prompt_templated,
#         "n_predict": n_predict,
#         "stop": ["### HUMAN:", "### ASSISTANT:", "HUMAN"],
#         "stream": "True"
#     }
#     try:
#         response = requests.post(sbc_host_url, headers=headers, data=json.dumps(data))
        
#         if response.status_code == 200:
#             return response.json()['content']
#         else:
#             response.raise_for_status()
#     except:
#         raise gr.Warning("Apologies for the inconvenience! Our model is currently self-hosted and unavailable at the moment.")


# def chatty(prompt, messages):
#     # print(prompt)
#     # print(f'messages: {messages}')
#     past_messages = ''
#     if len(messages) > 0:
#         for idx, message in enumerate(messages):
#             print(f'idx: {idx}, message: {message}')
#             past_messages += f'\n### HUMAN: {message[0]}'
#             past_messages += f'\n### ASSISTANT: {message[1]}'
                
                
#         # past_messages = messages[0][0]
#     # print(f'past_messages: {past_messages}')
#     messages = get_completion(prompt, past_messages)
#     return messages.split('### ASSISTANT:')[-1]

# stream
def chatty(prompt, messages, n_predict=128):
    # print(prompt)
    # print(f'messages: {messages}')
    past_messages = ''
    if len(messages) > 0:
        for idx, message in enumerate(messages):
            # print(f'idx: {idx}, message: {message}')
            past_messages += f'\n### HUMAN: {message[0]}'
            past_messages += f'\n### ASSISTANT: {message[1]}'
                
    system = "### System: You help to brainstorm ideas.\n"
    prompt_templated = f'{system} {messages}\n ### HUMAN:\n{prompt} \n ### ASSISTANT:'
    
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt_templated,
        "n_predict": n_predict,
        "stop": ["### HUMAN:", "### ASSISTANT:", "HUMAN"],
        "stream": True
    }

    result = ""
    try:
        response = requests.post(sbc_host_url, headers=headers, data=json.dumps(data), stream=True)
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        result += json.loads(line.decode('utf-8').replace('data: ', ''))['content']
                    except:
                        # LMStudio response has empty token 
                        pass
                    yield result
        else:
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise gr.Warning("Apologies for the inconvenience! Our model is currently self-hosted and unavailable at the moment.")


with gr.Blocks() as demo:
    gr.Image("sbc.jpg")
    gr.ChatInterface(
    fn=chatty,
    title="DevQuasar/llama3_8b_chat_brainstorm-GGUF on Orange Pi5 plus with llama.cpp",
    description="Brainstorm facilitates idea exploration through interaction with a Language Model (LLM). Rather than providing direct answers, the model engages in a dialogue with users, offering probing questions aimed at fostering deeper contemplation and consideration of various facets of their ideas."
) 


if __name__ == "__main__":
    demo.launch()