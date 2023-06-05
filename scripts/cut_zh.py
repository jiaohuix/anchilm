'''
中文分词，最终都转为简体

pip install jieba zhconv transformers jiayan kenlm
wget 'https://bj.bcebos.com/v1/ai-studio-online/ba7cc20a9f7a4a598cf8d5dd4bb8174cd7c0d587912c4d1d9b0a4775d3d4f1c0?responseContentDisposition=attachment%3B%20filename%3Djiayan_models.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-05-05T06%3A46%3A29Z%2F-1%2F%2Ff435166238eb5db107bcfe2bd88a4437e733c717fb4eb09a26eaee126f572495' -O jieyan_model.zip
unzip  jieyan_model.zip
from jiayan import load_lm
lm = load_lm('jiayan_models/jiayan.klm')
'''
import os
import sys
import jieba
import zhconv
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from transformers import AutoTokenizer
from jiayan import load_lm
from jiayan import CharHMMTokenizer

def cut_words(sent, backend="jieba", tokenizer=None):
    sent=sent.strip()
    if backend == "jieba":
        sent = zhconv.convert(sent, 'zh-hans')
        words = jieba.lcut(sent)
        # sent = zhconv.convert(" ".join(words), 'zh-hant') + "\n"
        sent = " ".join(words) + "\n"

    else:
        words = tokenizer.tokenize(sent)
        sent = zhconv.convert(" ".join(words) , 'zh-hans') + "\n"
    return sent

def main(infile,outfile,workers=1,backend="jieba",lm_file="jiayan.klm"):
    with open(infile,"r",encoding="utf-8") as fr,open(outfile,"w",encoding="utf-8") as fw:
        pool=Pool(processes=workers)
        tokenizer = None
        if backend == "jiayan":
            lm = load_lm(lm_file)
            tokenizer = CharHMMTokenizer(lm)
        elif backend == "siku":
            tokenizer = AutoTokenizer.from_pretrained("SIKU-BERT/sikuroberta")
        elif backend == "wwm":
            tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

        cut_words_fn= partial(cut_words, backend=backend, tokenizer=tokenizer)
        sentences = pool.imap(cut_words_fn,fr,chunksize=1000)
        for sent in tqdm(sentences):
            fw.write(sent)

if __name__ == '__main__':
    assert len(sys.argv)>=4,f"usage: python {sys.argv[0]} <infile> <outfile>  <workers>  <backend=jieba>(opt) <lm_file=jieyan.klm>(opt)"
    infile=sys.argv[1]
    outfile=sys.argv[2]
    workers=int(sys.argv[3])
    backend= sys.argv[4] if len(sys.argv)>=5 else "jieba"
    lm_file = sys.argv[5] if len(sys.argv)>=6 else "jiayan.klm"

    assert backend in ["jieba", "jiayan", "siku", "wwm"]
    main(infile,outfile,workers,backend,lm_file)
