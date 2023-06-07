echo "<data> <save> <src=de>(opt) <tgt=en>(opt)  <tokens=4096>(opt) <updates=60000>(opt) <ptm>(opt) "
DATA=$1
SAVE=$2
src=${3:-"gu"}
tgt=${4:-"zh"}
tokens=${5:-"8192"}
updates=${6:-"70000"}
PTM=${7:-"checkpoint_last.pt"}
epochs=50
mkdir -p $SAVE
if [ -f "$PTM" ]
then
finetune_args="--reset-lr-scheduler --reset-optimizer --reset-dataloader --restore-file $PTM"
else
finetune_args=""
fi


fairseq-train \
    $DATA --save-dir  $SAVE -s $src -t $tgt $finetune_args \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens $tokens --no-epoch-checkpoints --max-update $updates --max-epoch $epochs \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --fp16 \
    --tensorboard-logdir $SAVE/vislogs/   >> $SAVE/train.log
