import os
import gradio as gr
from openai import OpenAI
# from huggingface_hub import InferenceClient

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
# I can run it on the workstation with LMStudio and access it via the proxy
# client = InferenceClient("DevQuasar/llama3_8b_chat_brainstorm-v2.1")

self_host_url = os.environ['URL']
api_key = os.environ['API_KEY']

client = OpenAI(base_url=self_host_url, api_key=api_key)


def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message+" "}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    response = ""

    try:
        for message in client.chat.completions.create(
            model="DevQuasar/llama3_8b_chat_brainstorm-GGUF",
            messages=messages,
            temperature=temperature,
            stream=True,
        ):
            
            token = message.choices[0].delta.content
    
            try:
                response += token
            except:
                # LMStudio response has empty token 
                pass
            yield response
    except:
        raise gr.Warning("Apologies for the inconvenience! Our model is currently self-hosted and unavailable at the moment.")

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
title = "Brainstorm Demo by devquasar.com"
desc = "Please note that this model is self-hosted, which means it may not be available at all times. Thank you for your understanding!"
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant helps to brainstorm ideas.", label="System message"),
        #gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        #gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)"),
    ],
    title=title,
    description=desc
)


if __name__ == "__main__":
    demo.launch()