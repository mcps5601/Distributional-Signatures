#dataset=amazon
#data_path="data/amazon.json"
#n_train_class=10
#n_val_class=5
#n_test_class=9

# dataset=fewrel
# data_path="data/fewrel.json"
# n_train_class=65
# n_val_class=5
# n_test_class=10

# dataset=20newsgroup
# data_path="data/20news.json"
# n_train_class=8
# n_val_class=5
# n_test_class=7

dataset=banking77
data_path="data/banking77.json"

n_train_class=25
n_val_class=25
n_test_class=27


if [ "$dataset" = "fewrel" ]; then
    CUDA_LAUNCH_BLOCKING=1 python src/main.py \
        --cuda 0 \
        --way 5 \
        --shot 1 \
        --query 25 \
        --mode train \
        --embedding meta \
        --classifier r2d2 \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --auxiliary pos \
        --meta_iwf \
        --meta_w_target
else
    shot=1
    CUDA_LAUNCH_BLOCKING=1 python src/main.py >> ${dataset}_bert_baseline_${shot}s_new.txt \
        --cuda 0 \
        --way 5 \
        --shot $shot \
        --query 25 \
        --mode train \
        --embedding meta \
        --classifier r2d2 \
        --dataset=$dataset \
        --data_path=$data_path \
        --n_train_class=$n_train_class \
        --n_val_class=$n_val_class \
        --n_test_class=$n_test_class \
        --meta_iwf \
        --meta_w_target \
        --bert
fi
