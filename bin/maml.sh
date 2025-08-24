#dataset=amazon
#data_path='data/amazon.json'
#n_train_class=10
#n_val_class=5
#n_test_class=9

#dataset=fewrel
#data_path='data/fewrel.json'
#n_train_class=65
#n_val_class=5
#n_test_class=10

#dataset=20newsgroup
#data_path='data/20news.json'
#n_train_class=8
#n_val_class=5
#n_test_class=7

# dataset=huffpost
# data_path='data/huffpost.json'
# data_path="data/t5-large_huffpost_roberta-large-mnli_10N_top-k_40_EorN.json"
# n_train_class=20
# n_val_class=5
# n_test_class=16

dataset=banking77
# data_path="/data/workspace/Fewshot/data/banking77.json"
n_train_class=25
n_val_class=25
n_test_class=27

#dataset=rcv1
#data_path='data/rcv1.json'
#n_train_class=37
#n_val_class=10
#n_test_class=24

# dataset=reuters
# data_path='data/reuters.json'
# n_train_class=15
# n_val_class=5
# n_test_class=11
N=10
len=128
mode=E_only

for data_path in "/data/workspace/Fewshot/data/banking77/t5-large_banking77_${N}N_top-k_40_l${len}_roberta-large-mnli_${mode}.json"
# for data_path in "/data/workspace/Fewshot/data/banking77.json"
do
    for shot in 5
    do
        for embedding in idf
        do
            # python src/main.py >> maml_${embedding}_${shot}s_banking77_baseline_0707.txt \
            python src/main.py >> maml_${embedding}_${shot}s_banking77_${len}_${mode}_1070_0708.txt \
                --cuda 1 \
                --way 5 \
                --shot $shot \
                --query 25 \
                --mode 'train' \
                --embedding $embedding \
                --classifier 'mlp' \
                --dataset=$dataset \
                --data_path=$data_path \
                --n_train_class=$n_train_class \
                --n_val_class=$n_val_class \
                --n_test_class=$n_test_class \
                --maml \
                --maml_innersteps 10
        done
    done
done

for data_path in "/data/workspace/Fewshot/data/banking77/t5-large_banking77_${N}N_top-k_40_l${len}_roberta-large-mnli_${mode}.json"
do
    for shot in 1 5
    do
        for embedding in cnn avg
        do
            python src/main.py >> maml_${embedding}_${shot}s_banking77_${len}_${mode}_1070_0708.txt \
                --cuda 1 \
                --way 5 \
                --shot $shot \
                --query 25 \
                --mode 'train' \
                --embedding $embedding \
                --classifier 'mlp' \
                --dataset=$dataset \
                --data_path=$data_path \
                --n_train_class=$n_train_class \
                --n_val_class=$n_val_class \
                --n_test_class=$n_test_class \
                --maml \
                --maml_innersteps 10
        done
    done
done


for data_path in "/data/workspace/Fewshot/data/banking77.json"
do
    for shot in 5
    do
        for embedding in idf
        do
            python src/main.py >> maml_${embedding}_${shot}s_banking77_baseline_1070_0708.txt \
                --cuda 1 \
                --way 5 \
                --shot $shot \
                --query 25 \
                --mode 'train' \
                --embedding $embedding \
                --classifier 'mlp' \
                --dataset=$dataset \
                --data_path=$data_path \
                --n_train_class=$n_train_class \
                --n_val_class=$n_val_class \
                --n_test_class=$n_test_class \
                --maml \
                --maml_innersteps 10
        done
    done
done

for data_path in "/data/workspace/Fewshot/data/banking77.json"
do
    for shot in 1 5
    do
        for embedding in cnn avg
        do
            python src/main.py >> maml_${embedding}_${shot}s_banking77_baseline_1070_0708.txt \
                --cuda 1 \
                --way 5 \
                --shot $shot \
                --query 25 \
                --mode 'train' \
                --embedding $embedding \
                --classifier 'mlp' \
                --dataset=$dataset \
                --data_path=$data_path \
                --n_train_class=$n_train_class \
                --n_val_class=$n_val_class \
                --n_test_class=$n_test_class \
                --maml \
                --maml_innersteps 10
        done
    done
done