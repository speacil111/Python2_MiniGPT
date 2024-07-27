import requests
import json
url="https://p33279i881.vicp.fun/v1/chat/completions"  
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'sk-tjVLmu32DjZsgfUYDe0648Dc112e4dB2B0Cc71C029Ad7b97'  
}
def generate_sft_data():
    file = 'data/dataset/wiki-zh-subset-train_subset.jsonl'
    all_data=[]
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            all_data.append(data['text'])

    for index,prompt in enumerate(all_data):
        print(index)
        prompt+='''\n请你根据以上知识提出单个有意义的问题并做出不过短也不长的回答。以下是几个样例
        {"question": "桂城站的出入口及周边有哪些设施？", "answer": "桂城站目前设置的2个出入口均位于南桂东路南侧。本站设有便利店、面包糕饼店、中国银行自动柜员机、自动售货机及“好易”机。"}
        {"question": "模糊集的隶属函数是什么？", "answer": "模糊集的隶属函数是一个从论域到单位区间的映射，用来表示元素对该集的归属程度。"}
        请你严格仿照这样的类json格式输出一个，同时不要输出其他东西.'''
        payload = {
            "model":"glm-3-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }

        response = requests.post(url, headers=headers, json=payload,stream=True)
        if response.status_code == 200:
            json_response = json.loads(response.text)
            try:
                print(json_response['choices'][0]['message']['content'])
                dict = json.loads(json_response['choices'][0]['message']['content'])
                with open('sft_data_new.jsonl', 'a', encoding='utf-8') as outputf:
                    json.dump(dict, outputf, ensure_ascii=False)
                    outputf.write('\n')
            except Exception as e:
                print(f"发生错误: {e}") 
        else:
            print("请求失败：", response.content)

if __name__ == '__main__':
    generate_sft_data()