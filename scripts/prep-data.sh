# 1.数据划分  2.分词 3.bpe
bpe_ops=12000
## 1.DATA
DATA_URL="https://bj.bcebos.com/v1/ai-studio-online/ee18ac5bf26b476dafb97229692f0cfdd45ddff669774421b5dfa30fa8b7754a?responseContentDisposition=attachment%3B%20filename%3DEvaHan2023.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-06-05T11%3A13%3A33Z%2F-1%2F%2F715dfee070a19f696a57443a893a5a85aeb404e7600346b3c5aa029915c9cf02"
wget $DATA_URL -O EvaHan2023.zip
unzip EvaHan2023.zip
## jiayan model
JIAYAN_URL="https://bj.bcebos.com/v1/ai-studio-online/ba7cc20a9f7a4a598cf8d5dd4bb8174cd7c0d587912c4d1d9b0a4775d3d4f1c0?responseContentDisposition=attachment%3B%20filename%3Djiayan_models.zip&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2023-05-05T06%3A46%3A29Z%2F-1%2F%2Ff435166238eb5db107bcfe2bd88a4437e733c717fb4eb09a26eaee126f572495"
wget $JIAYAN_URL  -O jiayan_models.zip
unzip jiayan_models.zip

# 2.SPLIT
git clone https://gitee.com/miugod/nmt_data_tools
python nmt_data_tools/my_tools/train_dev_split.py gu zh EvaHan2023/train.24 . 250
python nmt_data_tools/my_tools/train_dev_split.py gu zh EvaHan2023/train.qin EvaHan2023/ 250
cat train.gu >> EvaHan2023/train.gu
cat train.zh >> EvaHan2023/train.zh
cat dev.gu >> EvaHan2023/dev.gu
cat dev.zh >> EvaHan2023/dev.zh
mv EvaHan2023/dev.gu EvaHan2023/valid.gu
mv EvaHan2023/dev.zh EvaHan2023/valid.zh

# 3.SPLIT
tok_dir="EvaHan2023/tok/"
mkdir -p $tok_dir
for prefix in train valid
  do
    python cut_zh.py EvaHan2023/$prefix.gu $tok_dir/$prefix.gu 6 jiayan jiayan_models/jiayan.klm
    python cut_zh.py EvaHan2023/$prefix.zh $tok_dir/$prefix.zh 6 jieba
  done


# 4.subword (optional)
bpe_dir="EvaHan2023/bpe/"
mkdir -p $bpe_dir

cat  EvaHan2023/tok/train.*  EvaHan2023/tok/valid.* > EvaHan2023/tok/all
subword-nmt learn-bpe -s $bpe_ops < EvaHan2023/tok/all  > $bpe_dir/codes
rm EvaHan2023/tok/all
# apply BPE
for prefix in train valid
 do
   subword-nmt apply-bpe -c $bpe_dir/codes < $tok_dir/$prefix.gu  > $bpe_dir/$prefix.gu
   subword-nmt apply-bpe -c $bpe_dir/codes < $tok_dir/$prefix.zh  > $bpe_dir/$prefix.zh
 done

# 5. vocab
cat $bpe_dir/train.* $bpe_dir/valid.* > $bpe_dir/all
python nmt_data_tools/my_tools/build_dictionary.py $bpe_dir/all
python nmt_data_tools/my_tools/json2dict.py $bpe_dir/all.json  $bpe_dir/dict.gu.txt
cp $bpe_dir/dict.gu.txt $bpe_dir/dict.zh.txt
rm $bpe_dir/all
