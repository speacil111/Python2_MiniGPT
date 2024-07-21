from flask import Flask, request, jsonify
from sample_gradio import generate_text_arena
import sample_gradio

app = Flask(__name__)
@app.route('/', methods=['POST'])
def generate():
    data = request.json
    start_text = data.get('start_text')
    temperature = data.get('temperature', 0.8)
    top_k = data.get('top_k', 200)
    max_new_tokens = int(data.get('max_new_tokens', 256))
    generated_text = generate_text_arena(start_text,max_new_tokens,temperature,top_k)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)


