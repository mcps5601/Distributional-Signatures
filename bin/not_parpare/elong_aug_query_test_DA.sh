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
data_path="data/huffpost.json"
# DA_path="data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_EorN.json"

n_train_class=20
n_val_class=5
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

generate='t5-large'
csv_path='elong_aug_query_test_DA'
for way_shot in '5way-1shot' '5way-5shot'
do
    if [ "$way_shot" = '5way-1shot' ]; then
        way=5
        shot=1
    elif [ "$way_shot" = '5way-5shot' ]; then
        way=5
        shot=5
    fi

    for data_path in 'data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_C_only.json' 'data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_EorN.json' 'data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_N_only.json'
    do
        r=0
        if [ "$DA_path" = "data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_C_only.json" ]; then
            DA_name="C_only"
        elif [ "$DA_path" = "data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_EorN.json" ]; then
            DA_name="EorN"
        elif [ "$DA_path" = "data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_N_only.json" ]; then
            DA_name="N_only"
        fi

        for seed in 42 80 100 200 300
        do
            ((r++))
            result_path='result/'$csv_path'_'$way_shot'_'$generate'_'$DA_name'_'$r

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
                    --cuda 0 \
                    --way=$way \
                    --shot=$shot \
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
                    --DA_vocab use_DA \
                    --DA_path $DA_path \
                    --aug_mode elongation \
                    --use_query_DA \
                    --test_DA \
                    --result_path=$result_path \
                    --csv_path=$csv_path \
                    --seed=$seed
            fi
        done
    done
done
