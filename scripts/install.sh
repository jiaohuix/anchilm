/usr/bin/python3.8 -m pip install --upgrade pip
# AttributeError: module 'distutils' has no attribute 'version'
pip install setuptools==59.5.0

conda create -n env python=3.7
conda activate env
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
# 官网的，慢点不容易出错2.0.1不会报s90的错误
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia


pip install torch==1.13.0+cu117 --extra-index-url -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/

git clone https://github.com/pytorch/fairseq 
cd fairseq
pip install --editable ./ -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install sacremoses tensorboardX   sacrebleu==1.5 apex     fastcore omegaconf jieba  sentencepiece -i https://pypi.tuna.tsinghua.edu.cn/simple/
