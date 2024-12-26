#!/bin/bash


gpu_id=0
exp_id=1
dataset_path='../datasets/ReDReS.json'
#dataset_path='../datasets/ReDAD.json'
#dataset_path_eval='../datasets/ReDReS.json'
dataset_path_eval='../datasets/ReDAD.json'
evaluation_on='alzheimer_s_disease'
#evaluation_on='rett_syndrome'
model_ids=(
  1
  2
  )
epoch=50
batch_size=16
embed_mode='PubMedBERT_base'
#embed_mode='PubMedBERT_large'
margin=0.0
threshold=0.5
lr=0.00001
dropout=0.3

aggregations=(
    'start_start'
    'end_end'
    'start_end_start_end'
    'ent_context_ent_context'
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
            output_dir='results_cross_disease/'${evaluation_on}'/model_'${model_id}'/exp_'${exp_id}'/'${embed_mode}'/'${eval_metric}'/'${aggregation}'/'
            seed=42
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=3
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=7
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=21
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=77
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=24
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=69
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=96
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=44
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            seed=11
            echo 'Seed: '${seed}
            output_file='seed_'${seed}
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset_path $dataset_path --dataset_path_eval $dataset_path_eval --do_train --do_eval --do_cross_disease_training --model_id $model_id --epoch $epoch --batch_size $batch_size --embed_mode $embed_mode --lr $lr  --dropout $dropout --seed $seed --margin $margin --threshold $threshold --output_dir $output_dir --output_file $output_file --sentence_wise_splits --aggregation $aggregation
            echo '--------------'
            echo '--------------'
            echo '###################################################################################################################'
        done
    done
done