# Preprocess/binarize the data
TEXT=${1:-"EvaHan2023/bpe"}
outdir=${2:-"EvaHan2023"}
  fairseq-preprocess --source-lang gu --target-lang zh --srcdict $TEXT/dict.gu.txt \
     --tgtdict $TEXT/dict.zh.txt  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/valid \
    --destdir data-bin/$outdir \
    --workers 20

