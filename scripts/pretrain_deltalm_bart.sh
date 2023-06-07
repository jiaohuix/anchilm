# span
DATA=$1
SAVE=$2
mkdir -p $SAVE
tokens=${3:-"2048"}
updates=${4:-"50000"}
reduce=${5:-"top1"}

bart_args="--task denoising --max-source-positions 1024 --max-target-positions 1024  --mask 0.35  --rotate 0.0 --mask-random 0.1 --permute-sentences 1.0 --insert 0.0 --poisson-lambda 3.5 --mask-length span-poisson --replace-length 1"

delta_args="--user-dir extension --arch deltalm_base --pretrained-checkpoint sikuroberta/pytorch_model.bin --vocab-map map_idx.json --reduce $reduce  --share-all-embeddings"
fairseq-train  $DATA  --save-dir $SAVE \
  $bart_args $delta_args \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 1.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens $tokens --update-freq 1 \
  --fp16 --max-update $updates --no-epoch-checkpoints \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --tensorboard-logdir $SAVE/vislogs/   >> $SAVE/train.log
