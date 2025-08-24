#dataset=amazon
#data_path="data/amazon.json"
#n_train_class=10
#n_val_class=5
#n_test_class=9

#dataset=fewrel
#data_path="data/fewrel.json"
#n_train_class=65
#n_val_class=5
#n_test_class=10

#dataset=20newsgroup
#data_path="data/20news.json"
#n_train_class=8
#n_val_class=5
#n_test_class=7

#dataset=huffpost
#data_path="data/huffpost.json"
#n_train_class=20
#n_val_class=5
#n_test_class=16

#dataset=rcv1
#data_path="data/rcv1.json"
#n_train_class=37
#n_val_class=10
#n_test_class=24

# dataset=reuters
# data_path="data/reuters.json"
# n_train_class=15
# n_val_class=5
# n_test_class=11

dataset=banking77
# data_path=/data/workspace/Fewshot/data/banking77.json
n_train_class=25
n_val_class=25
n_test_class=27

N=10
len=128
mode=E_only

for data_path in "/data/workspace/Fewshot/data/banking77/t5-large_banking77_${N}N_top-k_40_l${len}_roberta-large-mnli_${mode}.json"
do
    for embedding in cnn idf avg
    do
        if [ "$dataset" = "fewrel" ] && [ "$embedding" = "cnn" ]; then
            python src/main.py \
                --cuda 0 \
                --way 5 \
                --shot 1 \
                --query 25 \
                --mode "train" \
                --embedding $embedding \
                --classifier "proto" \
                --dataset=$dataset \
                --data_path=$data_path \
                --n_train_class=$n_train_class \
                --n_val_class=$n_val_class \
                --n_test_class=$n_test_class \
                --auxiliary pos
        else
            for shot in 1 5
            do
                python src/main.py >> proto_${embedding}_${shot}s_banking77_${len}_${mode}.txt\
                    --cuda 1 \
                    --way 5 \
                    --shot $shot \
                    --query 25 \
                    --mode "train" \
                    --embedding $embedding \
                    --classifier "proto" \
                    --dataset=$dataset \
                    --data_path=$data_path \
                    --n_train_class=$n_train_class \
                    --n_val_class=$n_val_class \
                    --n_test_class=$n_test_class
            done
        fi
    done
done
