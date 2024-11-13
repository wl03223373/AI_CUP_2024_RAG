import os
import json
import argparse
from tqdm import tqdm
from FlagEmbedding import FlagReranker, BGEM3FlagModel
import re
import statistics
import jieba

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

def load_data(source_path):
    """載入TXT版本的文字資料."""
    masked_file_ls = os.listdir(source_path)  
    corpus_dict = {int(file.replace('.txt', '')): read_txt(os.path.join(source_path, file)) for file in tqdm(masked_file_ls)}  # 讀取每個文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
    return corpus_dict

def read_txt(file_loc):
    """讀取txt檔案."""
    with open(file_loc, "r", encoding='utf-8') as file:
        content = file.read()
    return content

def bge_rerank_retrieve(qs, source, corpus_dict, category):
    """基於 BAAI/bge-reranker-v2-m3 模型進行檢索."""
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    scores = []
    highest_score = -10000

    is_split = False if category == 'faq' else True

    # You can map the scores into 0-1 by set "normalize=True", which will apply sigmoid function to the score
    for item in filtered_corpus:
        score = reranker.compute_score([qs, item])
        highest_score = score[0]

        for sub in split_sequence_with_overlap(item,2000,400):
            score = reranker.compute_score([qs, sub])
            if score[0] > highest_score:
                highest_score = score[0]

        for sub in split_sequence_with_overlap(item,1000,200):
            score = reranker.compute_score([qs, sub])
            if score[0] > highest_score:
                highest_score = score[0]
        
        for sub in split_sequence_with_overlap(item,500,100):
            score = reranker.compute_score([qs, sub])
            if score[0] > highest_score:
                highest_score = score[0]

        for sub in split_sequence_with_overlap(item,100,20):
            score = reranker.compute_score([qs, sub])
            if score[0] > highest_score:
                highest_score = score[0]

        scores.append(highest_score)
        highest_score = -10000

    # print(score) # 0.003497010252573502
    # print(scores)
    # 找到最相似的文檔
    # best_match_idx = score.argmax().item()
    print('分數: ' + str(max(scores)))
    print(scores)
    best_match_idx = scores.index(max(scores))
    best_match_doc = filtered_corpus[best_match_idx]

    # 找回與最佳匹配文本相對應的檔案名
    res = [key for key, value in corpus_dict.items() if value == best_match_doc]
    print(res[0])
    return res[0]  # 回傳檔案名

def split_sequence_with_overlap(text, size=500, overlap=100):
    """將文本進行有重疊的切塊."""
    # Initialize an empty list to hold the chunks
    chunks = []
    
    # Calculate the step size, which is the size minus the overlap
    step = size - overlap
    
    # Loop through the text in steps of the calculated step size
    for i in range(0, len(text), step):
        # Slice the text from the current index to the current index + size
        chunk = text[i:i + size]
        chunks.append(chunk)
    
    return chunks

from tqdm import tqdm

# 主程式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    # parser.add_argument('--question_path', type=str, default=r'D:\AIcup\競賽資料集\dataset\preliminary\questions_example.json', help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--question_path', type=str, default=r'D:\AIcup\競賽資料集\dataset\preliminary\questions_preliminary.json', help='讀取發布題目路徑')  # 問題文件的路徑
    parser.add_argument('--source_path', type=str, default=r'D:\AIcup\競賽資料集\reference', help='讀取參考資料路徑')  # 參考資料的路徑
    parser.add_argument('--output_path', type=str, default=r'C:\Users\Jason\iCloudDrive\研究所\aicup2024\pred\110901_pre.json', help='輸出符合參賽格式的答案路徑')  # 答案輸出的路徑

    args = parser.parse_args() 
    answer_dict = {"answers": []} 

    # 讀取問題檔案
    with open(args.question_path, 'rb') as f:
        qs_ref = json.load(f)

    # 載入參考資料
    source_path_insurance = os.path.join(args.source_path, 'insurance_text')
    corpus_dict_insurance = load_data(source_path_insurance)
    source_path_finance = os.path.join(args.source_path, 'finance_text')
    corpus_dict_finance = load_data(source_path_finance)

    # 讀取 FAQ 資料
    with open(os.path.join(args.source_path, 'faq/pid_map_content.json'), 'rb') as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    # 使用 tqdm 顯示問題檢索進度
    for q_dict in tqdm(qs_ref['questions'], desc="處理問題"):
        # if q_dict['qid'] != 86:
        #     continue
        if q_dict['category'] == 'finance':
            retrieved = bge_rerank_retrieve(q_dict['query'], q_dict['source'], corpus_dict_finance, q_dict['category'])
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'insurance':
            retrieved = bge_rerank_retrieve(q_dict['query'], q_dict['source'], corpus_dict_insurance, q_dict['category'])
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        elif q_dict['category'] == 'faq':
            corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items() if key in q_dict['source']}
            retrieved = bge_rerank_retrieve(q_dict['query'], q_dict['source'], corpus_dict_faq, q_dict['category'])
            answer_dict['answers'].append({"qid": q_dict['qid'], "retrieve": retrieved})

        else:
            raise ValueError("Unknown category encountered")
        

    # 將答案字典保存為 json 文件
    with open(args.output_path, 'w', encoding='utf8') as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)
