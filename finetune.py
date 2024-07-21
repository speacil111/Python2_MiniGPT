
# import json

# from zhipuai import ZhipuAI
# # 假设这是你的 ChatGLM API 密钥

# client = ZhipuAI(api_key='90cb70a2e18674d1f71b47662bcc2920.CX8cy22P57f04lJu')


# def generate_question_answer():
#     response = client.chat.completions.create(
#                                         model="glm-3-turbo",
#                                         messages=[{"role": "user", "content": "随机生成一个常识的中文问答"}],
#                                         temperature=0.8,
#                                         top_p=0.7,
#                                         max_tokens=200,)
 
#     text = response['choices'][0]['text'].strip().split('\n')
#     question = text[0].strip()
#     answer = text[1].strip()
#     return {"question": question, "answer": answer}

# def generate_data(num_records):
#     data = [generate_question_answer() for _ in range(num_records)]
#     return data


# num_records = 20
# data = generate_data(num_records)

# with open('chatglm_data.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

import openai
import json
import time
from openai import OpenAI
from random import randint
client = OpenAI(
    api_key="sk-ZibwPvsw0gmNCjmqcOh8UIjuhCmn7sSQ7gUuvlgthmM4ja4O",
    base_url="https://api.moonshot.cn/v1",
)
def generate_question_answer():

    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "user", "content": "生成一个中文问答,总字数在100字左右."}
        ],
        temperature=0.8,
        max_tokens=200,

    )

    text = completion.choices[0].message.content.strip().split('\n')
    question = text[0].strip()
    answer = text[1].strip()
    return {"question": question, "answer": answer}

num_records = 20  # Generate 20 records for demonstration

with open('finetune_data.json', 'w', encoding='utf-8') as f:
    for i in range(num_records):
        time.sleep(randint(1, 2))
        data=generate_question_answer()
        json_str = json.dumps(data, ensure_ascii=False)+'\n'
        f.write(json_str)
    f.close()