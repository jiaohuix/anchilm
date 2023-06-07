DATA=$1
CKPT=$2
fairseq-generate $DATA \
    --path $CKPT \
    --beam 5 --batch-size 128 --remove-bpe
