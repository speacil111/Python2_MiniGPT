import gradio as gr
import requests
import time
import threading
from gradio_client import Client
def call_api(url, input_text, max_new_tokens,temperature, top_k):
    data = {
        "start_text": input_text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k, 
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        text = result['generated_text']
        return text
    except requests.exceptions.RequestException as e:
        return f"API 调用失败: {e}"

def generate_response(model1, model2, input_text,  max_new_tokens,temperature, top_k):
    model_urls = {
        "运沛然的海螺": "http://183.173.120.60:5000/",
        "颜子俊的海螺": "https://d6ca7897831e13aea0.gradio.live/",
        "魏来的海螺": "https://f65e3c7409b8d061fd.gradio.live/"
    }
    response1 = [""]
    response2 = [""]

    def fetch_response1():
        print(model1)
        if model1 not in model_urls:
            response1[0] = "请选择海螺1"
        elif model1 == "魏来的海螺":
            client1 = Client(model_urls[model1])
            output1 = client1.predict(input_text=input_text, api_name="/predict")
            response1[0]=output1
        elif model1 == "颜子俊的海螺":
            client1 = Client(model_urls[model1])
            output1 = client1.predict(start=input_text, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, repetition_penalty=1.2, api_name="/predict")
            response1[0] = output1
        elif model1 == "运沛然的海螺":
            response1[0] = call_api(model_urls[model1], input_text,  max_new_tokens,temperature, top_k)
    def fetch_response2():
        print(model2)
        if model2 not in model_urls:
            response2[0] = "请选择海螺2"
        elif model2 == "魏来的海螺":
            client1 = Client(model_urls[model2])
            output1 = client1.predict(input_text=input_text, api_name="/predict")
            response2[0]=output1
        elif model2 == "颜子俊的海螺":
            client1 = Client(model_urls[model2])
            output1 = client1.predict(start=input_text, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, repetition_penalty=1.2, api_name="/predict")
            response2[0] = output1
        elif model2 == "运沛然的海螺":
            response2[0] = call_api(model_urls[model2], input_text,  max_new_tokens,temperature, top_k)
    thread1 = threading.Thread(target=fetch_response1)
    thread2 = threading.Thread(target=fetch_response2)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
    full_response1 = response1[0]
    full_response2 = response2[0]

    max_len = max(len(full_response1), len(full_response2))

    for i in range(max_len):
        partial_response1 = full_response1[:i+1] if i < len(full_response1) else full_response1
        partial_response2 = full_response2[:i+1] if i < len(full_response2) else full_response2
        time.sleep(0.05)
        yield partial_response1, partial_response2

model_options = ["运沛然的海螺", "颜子俊的海螺", "魏来的海螺"]

iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Dropdown(model_options, label="选择海螺1"),
        gr.Dropdown(model_options, label="选择海螺2"),
        gr.Textbox(label="输入问题", lines=3,placeholder="战斗爽"),
        gr.Slider(10, 512, step=1, value=256, label="Max New Tokens"),
        gr.Slider(0.1, 1.2, step=0.01, value=0.9, label="Temperature"),
        gr.Slider(10, 100, step=1, value=40, label="Top K"),
    ],
    outputs=[
        gr.Textbox(label="神奇海螺1",lines=7),
        gr.Textbox(label="神奇海螺2",lines=7),
    ],
    title="神奇海螺竞技场！",
    allow_flagging='never',
    theme='soft')

if __name__=="__main__":
    iface.launch()
