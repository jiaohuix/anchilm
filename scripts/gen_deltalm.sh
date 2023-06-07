DATA=$1
CKPT=$2
fairseq-generate $DATA --user-dir extension --arch deltalm_base \
    --path $CKPT \
    --beam 5 --batch-size 128 --remove-bpe
