from keybert import KeyBERT
from konlpy.tag import Okt
import re
okt = Okt()

kw_model = KeyBERT(model="all-MiniLM-L6-v2")

stop_word = ['있다']

def keybert_extraction(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1,1), top_n=10)
    result_keyword = []
    for i in keywords:
        result_keyword.append(i[0])

    return result_keyword

def input_preprocesing(input): # list
    result_list = []
    result = okt.pos(input, stem=True, norm=True)
    for j in result:
        if j[0] not in result_list and j[1] in ['Adjective', 'Noun', 'Verb', 'Alpha'] and j[0] not in stop_word:
            result_list.append(j[0])
    
    return result_list

def clean_str(text):
    pattern = '(http|ftp|https)://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '([ㄱ-ㅎㅏ-ㅣ]+)'  # 한글 자음, 모음 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '<[^>]*>'         # HTML 태그 제거
    text = re.sub(pattern=pattern, repl='', string=text)
    pattern = '[^\w\s\n]'         # 특수기호제거
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]',' ', string=text)
    text = re.sub('\n', '.', string=text)
    return text 

def str_to_list_keyword(text):
    result = clean_str(text)
    result_list = keybert_extraction(result)
    return_list = []
    for i in result_list:
        extraction_list = input_preprocesing(i)
        for j in extraction_list:
            if j not in return_list:
                return_list.append(j)
    
    return(return_list)

def list_to_list_keyword(text_list):
    result_list = []
    for i in text_list:
        result = input_preprocesing(i)
        for j in result:
            if j not in result_list:
                result_list.append(j)

    return result_list