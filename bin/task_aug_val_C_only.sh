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
# data_path="data/task_aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_C_only.json"

n_train_class=20
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

generate='nli-generator'
csv_path='task_aug_all'
for way_shot in '5way-1shot' '5way-5shot'
do
    if [ "$way_shot" = '5way-1shot' ]; then
        way=5
        shot=1
    elif [ "$way_shot" = '5way-5shot' ]; then
        way=5
        shot=5
    fi

    # for data_path in 'data/aug_all_roberta_select_huffpost_G1_10N_top-k_40.json' "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_C_only.json" "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_N_only.json" "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_EorN.json"
    for data_path in "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_C_only.json" "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_N_only.json" "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_EorN.json"
    do
        r=0
        if [ "$data_path" = "data/aug_all_roberta_select_huffpost_G1_10N_top-k_40.json" ]; then
            DA_name="C_only"
        elif [ "$data_path" = "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_C_only.json" ]; then
            DA_name="C_only"
        elif [ "$data_path" = "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_EorN.json" ]; then
            DA_name="EorN"        
        elif [ "$data_path" = "data/aug_all_t5-large_huffpost_roberta-large-mnli_10N_top-k_40_N_only.json" ]; then
            DA_name="N_only"
        fi

        for seed in 42 80 100 200 300
        do
            ((r++))
            if [ "$DA_name" = "double_text" ]; then
                result_path='result/'$csv_path'_'$way_shot'_'$DA_name'_'$r
            else
                result_path='result/'$csv_path'_'$way_shot'_'$generate'_'$DA_name'_'$r
            fi

            if [ "$dataset" = "fewrel" ]; then
                python src/main.py \
                    --cuda 1 \
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
                    --cuda 1 \
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
                    --aug_mode task \
                    --task_aug_target val \
                    --task_aug_exclude_val_query \
                    --result_path=$result_path \
                    --csv_path=$csv_path \
                    --seed=$seed
            fi
        done
    done
done