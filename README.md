# anchilm

Ancient and Chinese lm



## 1 数据

ancientlm

```shell
pip install jieba zhconv transformers jiayan kenlm subword-nmt
# data
bash scripts/prep-data.sh
bash scripts/bin.sh
# eval
bash scripts/eval.sh <testfile> <genfile>
```



tensorboard:

```shell
#启动TensorboardX前，需要先关闭官方镜像中安装的tensorboard
~# supervisord ctl stop tensorboard

#启动TensorBoardX也通过tmux程序托管运行
~# tmux new -s tensorboard
~# tensorboard --logdir runs --host 0.0.0.0
```

## 2 训练

### 2.1 baseline

直接使用iwslt_de_en的arch或者base, 并且共享所有embeddings。

```shell
# iwslt_de_en 47m
bash scripts/train_nmt.sh  data-bin/EvaHan2023  ckpt/base/ gu zh 8192 60000 
# transformer base 57m
bash scripts/train_nmt_base.sh  data-bin/EvaHan2023  ckpt/base/ gu zh 8192 60000 
```

评估：

```shell
bash scripts/gen.sh  data-bin/EvaHan2023  ckpt/base/checkpoint_best.pt > gen_base.txt
tail gen_base.txt  # bleu=28.31 (iwslt_de_en)
```

### 2.2 rdrop

使用rdrop的loss增强。

```shell
# extension
git clone https://github.com/jiaohuix/fairseq_extension.git
cp -r fairseq_extension/extension .
# iwslt_de_en 47m
bash scripts/train_rdrop.sh  data-bin/EvaHan2023  ckpt/rdrop/ gu zh 8192 60000 
```

评估：

```shell
bash scripts/gen.sh  data-bin/EvaHan2023  ckpt/rdrop/checkpoint_best.pt > gen_rdrop.txt
tail gen_rdrop.txt  # bleu=29.05 (iwslt_de_en)
```

关键参数：



### 2.3 bpe drop

使用bpe drop对数据进行增强。

```shell
# 数据增强，iter=3扩大三倍, EvaHan2023/bpe/ -> data-bin/EvaHan2023_bpedp
bash scripts/prep_bpedp.sh
# 训练
bash scripts/train_nmt.sh  data-bin/EvaHan2023_bpedp  ckpt/bpedp/ gu zh 8192 60000 
```

评估：

```shell
bash scripts/gen.sh  data-bin/EvaHan2023_bpedp  ckpt/bpedp/checkpoint_best.pt > gen_bpedp.txt
tail gen_bpedp.txt  # bleu=28.98 (iwslt_de_en)
```

### 2.4 span corruption

使用文本片段破坏预训练任务2w step，再微调6w step翻译训练。

```shell
# extension
git clone https://github.com/jiaohuix/fairseq_extension.git
cp -r fairseq_extension/examples/span_mask_lm/  .
# dir: fairseq/anchilm/
cp span_mask_lm/scripts/span_mask_tokens_dataset.py ../fairseq/data/
cp span_mask_lm/scripts/span_masked_lm.py ../fairseq/tasks/

# lm data 	(EvaHan2023/lm_data,data-bin/lm_data)
bash scripts/lm_data.sh 

# pretrain
bash span_mask_lm/scripts/pretrain_span.sh data-bin/lm_data ckpt/span_lm 16384 20000 0.15 3

# finetune
bash span_mask_lm/scripts/train_nmt.sh  data-bin/EvaHan2023 checkpoints/span_lm_ft ckpt/span_lm/ckpt.pt gu zh 8192 60000

```

评估:

```shell
bash span_mask_lm/scripts/eval.sh data-bin/EvaHan2023 ckpt/span_lm_ft/checkpoint_best.pt  > gen_span.txt
tail gen_span.txt  # bleu=29.8 (iwslt_de_en)
```



### 2.5 denoise 

```shell
# extension
git clone https://github.com/jiaohuix/fairseq_extension.git
cp -r fairseq_extension/examples/bart/  .
# dir: fairseq/anchilm/
cp bart/scripts/denoising.py ../fairseq/tasks/

# lm data 	(EvaHan2023/lm_data,data-bin/lm_data)
bash scripts/lm_data.sh 

# pretrain
bash bart/scripts/pretrain_bart.sh data-bin/lm_data ckpt/bart_lm 8192 20000 

# finetune (bart微调可能词表要加一个mask token)
# cp -r data-bin/EvaHan2023 data-bin/EvaHan2023_bart
# echo  "<mask> 1" >> data-bin/EvaHan2023_bart/dict.zh.txt
# echo  "<mask> 1" >> data-bin/EvaHan2023_bart/dict.gu.txt
bash bart/scripts/train_nmt.sh  data-bin/EvaHan2023 ckpt/bart_lm_ft ckpt/bart_lm/checkpoint_best.pt gu zh 8192 60000

```

评估:

```shell
bash bart/scripts/eval.sh data-bin/EvaHan2023 ckpt/bart_lm_ft/checkpoint_best.pt  > gen_bart.txt
tail gen_bart.txt  # bleu=29.57 (iwslt_de_en)
```

关键参数:

```shell
--task denoising --max-source-positions 1024 --max-target-positions 1024  --mask 0.35  --rotate 0.0 --mask-random 0.1 --permute-sentences 1.0 --insert 0.0 --poisson-lambda 3.5 --mask-length span-poisson --replace-length 1 #https://github.com/facebookresearch/fairseq/issues/1899
```



### 2.6 deltalm

使用sikuroberta初始化编解码器参数,解码器使用deltalm的交替初始化方式, 并重新构建子词词表, 用对应sikuroberta的字粒度的向量top1来初始化新的词表(不做预训练,后面2.7-8先预训练再翻译微调)

```shell
# 下载权重
apt-get install git-lfs
git lfs clone https://huggingface.co/SIKU-BERT/sikuroberta
# 词表映射
echo -e "<s>\n</s>\n<pad>\n<unk>" > dict.txt
cut -f1 -d" " EvaHan2023/bpe/dict.gu.txt >> dict.txt
python fairseq_extension/examples/deltalm/map_vocab.py sikuroberta/vocab.txt dict.txt 
# write to unk.txt success, total 254 lines.

# 训练：(用的是EvaHan2023_bpedp)
bash scripts/train_deltalm.sh  data-bin/EvaHan2023/ ckpt/deltalm_top1 gu zh 8192 30000 top1

# 其他: reduce=sum
bash scripts/train_deltalm.sh  data-bin/EvaHan2023/ ckpt/deltalm_sum gu zh 8192 30000 sum

```

评估:

```shell
bash scripts/gen_deltalm.sh data-bin/EvaHan2023/ ckpt/deltalm_top1/checkpoint_best.pt  > gen_delta.txt
tail gen_delta.txt  # bleu=29.52 (deltalm base)
```

关键参数:

```shell
--user-dir extension --arch deltalm_base --pretrained-checkpoint sikuroberta/pytorch_model.bin --vocab-map map_idx.json --reduce top1 --lm-head cls.predictions.decoder
```

### 2.7 deltalm+span

```shell
# pretrain
bash scripts/pretrain_deltalm_span.sh data-bin/lm_data/ ckpt/deltalm_span_lm/ 8192 20000 0.15 3 top1

# finetune 
bash scripts/fine_deltalm_span.sh data-bin/EvaHan2023/ ckpt/deltalm_span_ft ckpt/deltalm_span_lm/checkpoint_best.pt gu zh 8192 60000
```



### 2.8 deltalm + denoise

```shell
# pretrain
bash scripts/pretrain_deltalm_bart.sh data-bin/lm_data/ ckpt/deltalm_bart_lm/ 8192 20000 top1	

# finetune 
bash scripts/fine_deltalm_bart.sh data-bin/EvaHan2023/ ckpt/deltalm_span_ft ckpt/deltalm_bart_lm/checkpoint_best.pt gu zh 8192 60000
```



6/8

