# -*- coding: utf-8 -*-
import openai

openai.api_key = '{open ai key}'

def chat_gpt(input):
    
    prompt = input + '\n 위 글을 의미있는 단어로 추출해줘'
    
    completion = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {'role': 'user', 'content': prompt}
        ],
    temperature = 0 
    )
    return completion['choices'][0]['message']['content']
