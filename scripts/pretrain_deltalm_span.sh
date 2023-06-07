# span
DATA=$1
SAVE=$2
mkdir -p $SAVE
tokens=${3:-"2048"}
updates=${4:-"15000"}
density=${5:-"0.15"}
length=${6:-"3"}
reduce=${7:-"top1"}

delta_args="--user-dir extension --arch deltalm_base --pretrained-checkpoint sikuroberta/pytorch_model.bin --vocab-map map_idx.json --reduce $reduce  --share-all-embeddings"

fairseq-train --task span_masked_lm --noise-density $density --mean-noise-span-length $length \
  $DATA  --save-dir $SAVE $delta_args \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 1.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens $tokens --update-freq 4 \
  --fp16 --max-update $updates --no-epoch-checkpoints \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --tensorboard-logdir $SAVE/vislogs/   >> $SAVE/train.log
