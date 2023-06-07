cp -r EvaHan2023/bpe/ EvaHan2023/bpedp/
python nmt_data_tools/my_tools/bpe_dropout.py EvaHan2023/tok/train.zh  EvaHan2023/bpedp/tmp.zh EvaHan2023/bpe/codes 3 0.1
python nmt_data_tools/my_tools/bpe_dropout.py EvaHan2023/tok/train.gu  EvaHan2023/bpedp/tmp.gu EvaHan2023/bpe/codes 3 0.1

# 去除空行
paste EvaHan2023/bpedp/tmp.gu EvaHan2023/bpedp/tmp.zh > EvaHan2023/bpedp/tmp
grep -v "ErrorTokenize" EvaHan2023/bpedp/tmp > EvaHan2023/bpedp/tmp.ok
cut -f 1  EvaHan2023/bpedp/tmp.ok >  EvaHan2023/bpedp/train.gu
cut -f 2  EvaHan2023/bpedp/tmp.ok >  EvaHan2023/bpedp/train.zh
rm EvaHan2023/bpedp/tmp*

# 二值化
bash scripts/bin.sh EvaHan2023/bpedp/ data-bin/EvaHan2023_bpedp
