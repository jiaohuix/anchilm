indir=${1:-"EvaHan2023/bpe"}
TEXT=${2:-"EvaHan2023/lm_data"}
mkdir $TEXT
cat $indir/train.* > $TEXT/train
cat $indir/valid.* > $TEXT/valid
cat $indir/valid.* > $TEXT/test

fairseq-preprocess \
   --only-source --srcdict $indir/dict.gu.txt \
   --trainpref $TEXT/train \
   --validpref $TEXT/valid \
   --testpref $TEXT/test \
   --destdir data-bin/lm_data \
   --workers 20
