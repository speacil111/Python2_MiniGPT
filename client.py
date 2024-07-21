import gradio as gr
import requests
import time
import threading

def call_api(url, input_text, temperature, top_k, max_new_tokens):
    data = {
        "start_text": input_text,
        "temperature": temperature,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        text = result['generated_text']
        return text
    except requests.exceptions.RequestException as e:
        return f"API 调用失败: {e}"

def generate_response(model1, model2, input_text, temperature, top_k, max_new_tokens):
    model_urls = {
        "运沛然的海螺1": "http://183.172.185.186:5000/",
        "运沛然的海螺2": "http://183.172.185.186:5000/",
        "运沛然的海螺3": "http://183.172.185.186:5000/"
    }
    response1 = [""]
    response2 = [""]

    def fetch_response1():
        if model1 not in model_urls:
            response1[0] = "请选择海螺1"
        else:
            response1[0] = call_api(model_urls[model1], input_text, temperature, top_k, max_new_tokens)

    def fetch_response2():
        if model2 not in model_urls:
            response2[0] = "请选择海螺2"
        else:
            response2[0] = call_api(model_urls[model2], input_text, temperature, top_k, max_new_tokens)
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

model_options = ["运沛然的海螺1", "运沛然的海螺2", "运沛然的海螺3"]

iface = gr.Interface(
    fn=generate_response,
    inputs=[
        gr.Dropdown(model_options, label="选择海螺1"),
        gr.Dropdown(model_options, label="选择海螺2"),
        gr.Textbox(label="输入问题", lines=3,placeholder="战斗爽"),
        gr.Slider(10, 512, step=1, value=256, label="Max New Tokens"),
        gr.Slider(0.1, 1, step=0.01, value=0.8, label="Temperature"),
        gr.Slider(10, 256, step=1, value=200, label="Top K"),
    ],
    outputs=[
        gr.Textbox(label="神奇海螺1",lines=7),
        gr.Textbox(label="神奇海螺2",lines=7),
    ],
    title="神奇海螺竞技场！",
    allow_flagging='never',
    theme='soft')

iface.launch()
