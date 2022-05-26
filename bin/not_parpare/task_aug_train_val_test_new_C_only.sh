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

# dataset=20newsgroup
# data_path="data/20news.json"
# n_train_class=8
# n_val_class=5
# n_test_class=7

dataset=huffpost
# data_path="data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_C_only.json"

n_train_class=40
n_val_class=10
n_test_class=16

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

for way_shot in '5way_1shot' '5way_5shot'
do
    for data_path in "data/aug_all_roberta_select_huffpost_G1_10N_top-k_40.json"
    do
        r=0
        if [ "$data_path" = "data/aug_all_roberta_select_huffpost_G1_10N_top-k_40.json" ]; then
            generate='nli-gernerator'
        elif [ "$data_path" = "pass" ]; then
            generate='t5-large'
        fi

        for seed in 42 80 100 200 300
        do
            ((r++))
            result_path='result/task_aug_train_val_test_new_C_only_'$way_shot'_'$generate'_'$r
        
            if [ "$dataset" = "fewrel" ]; then
                python src/main.py \
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
                python src/main.py \
                    --cuda １ \
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
                    --meta_iwf \
                    --meta_w_target \
                    --aug_mode task \
                    --task_aug_target train_val \
                    --test_new_only \
                    --result_path=$result_path \
                    --seed=$seed
            fi
        done
    done
done
