#!/bin/bash


gpu_id=0
exp_id=1
dataset_path='essential_dataset_total.json'
model_id=2
epoch=30
batch_size=16
embed_mode='PubMedBERT_base'
exp_setting='binary'
eval_metric='micro'
lr=0.00001
dropout=0.3
seed=42


aggregations=(
    'start_start'
    'end_end'
    'cls_start_start'
    'cls_end_end'
    'start_inter_start'
    'end_inter_end'
    'start_end_start_end'
    'cls_start_end_start_end'
    'start_end_inter_start_end'
)

cd ..
cd ..

for aggregation in "${aggregations[@]}"; do
    echo '##########################################'
    echo 'Aggregation: '${aggregation}
    echo '##########################################'
    output_dir='results/model_'${model_id}'/exp_'${exp_id}'/'${embed_mode}'/'${exp_setting}'/'${eval_metric}'/'${aggregation}'/'
    echo 'Fold: 0'
    output_file='fold_0'
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 0 --aggregation $aggregation
    echo '--------------'
    echo '--------------'
    echo 'Fold: 1'
    output_file='fold_1'
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 1 --aggregation $aggregation
    echo '--------------'
    echo '--------------'
    echo 'Fold: 2'
    output_file='fold_2'
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 2 --aggregation $aggregation
    echo '--------------'
    echo '--------------'
    echo 'Fold: 3'
    output_file='fold_3'
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 3 --aggregation $aggregation
    echo '--------------'
    echo '--------------'
    echo 'Fold: 4'
    output_file='fold_4'
    CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --do_train --do_eval --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --exp_setting $exp_setting --eval_metric $eval_metric --lr $lr  --dropout $dropout --seed $seed --output_dir $output_dir --output_file $output_file --sentence_wise_splits --fold 4 --aggregation $aggregation
    echo '--------------'
    echo '--------------'
    echo '###################################################################################################################'
done