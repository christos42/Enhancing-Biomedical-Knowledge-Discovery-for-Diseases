#!/bin/bash

gpu_id=0
exp_id=1
dataset_path='../../datasets/ReDReS.json'
#dataset_path='../../datasets/ReDAD.json'
model_ids=5
epoch=50
batch_size=16
embed_mode='PubMedBERT_base'
#embed_mode='PubMedBERT_large'
exp_setting='binary'
#exp_setting='multi_class'
eval_metric='micro'
#eval_metric='macro'
lr=0.00001
dropout=0.0
seed=42

attention_heads=(
 0
 1
 2
 3
 4
 5
 6
 7
 8
 9
 10
 11
 #12
 #13
 #14
 #15
)

aggregations=(
  'head_specific'
)

cd ..

for model_id in "${model_ids[@]}"; do
    echo '##########################################'
    echo 'Model: '${model_id}
    echo '##########################################'
    for aggregation in "${aggregations[@]}"; do
        echo '##########################################'
        echo 'Aggregation: '${aggregation}
        echo '##########################################'
        for attention_head in "${attention_heads[@]}"; do
            echo '##########################################'
            echo 'Attention head: '${attention_head}
            echo '##########################################'
            output_dir='results/rett_syndrome/model_'${model_id}'_'${aggregation}'/exp_'${exp_id}'/'${embed_mode}'/'${exp_setting}'/'${eval_metric}'/attention_head_'${attention_head}'/'
            #output_dir='results/alzheimer_s_disease/model_'${model_id}'_'${aggregation}'/exp_'${exp_id}'/'${embed_mode}'/'${exp_setting}'/'${eval_metric}'/attention_head_'${attention_head}'/'
            echo 'Fold: 0'
            output_file='fold_0'
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 0 --attention_head $attention_head  --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            echo 'Fold: 1'
            output_file='fold_1'
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 1 --attention_head $attention_head  --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            echo 'Fold: 2'
            output_file='fold_2'
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 2 --attention_head $attention_head  --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            echo 'Fold: 3'
            output_file='fold_3'
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 3 --attention_head $attention_head  --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            echo 'Fold: 4'
            output_file='fold_4'
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 4 --attention_head $attention_head  --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            echo '###################################################################################################################'
        done
    done
done