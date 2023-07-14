# 

# Correlation Detection
This subdirectory consisted of the supervised approaches of the study.

---

## Setup
### Requirements
 - Python 3.5+
 - pytorch (tested with version 2.0.1)
 - scikit-learn (tested with version 1.2.2)
 - tqdm (tested with version 4.65.0)
 - transformers (tested with version 4.29.2)
 - numpy (tested with version 1.23.5)

---

### Execution steps
 - Run ```main.py --dataset_path [--dataset_path_eval] --do_train --do_eval [--do_cross_disease_training] --model_id 
   --epoch --batch_size --eval_batch_size --embed_mode --exp_setting --eval_metric --lr --weight_decay --dropout 
   --seed --output_dir --output_file --sentence_wise_splits --fold --aggregation``` to train and evaluate the models. 
   Arguments:
   - dataset_path (string): the path to the dataset
   - dataset_path_eval (string) (optional): the to the evaluation dataset in case that the training and evaluation 
     datasets are different files. The argument is used when cross-disease training and evaluation is applied.
   - do_train (store_value): boolean value to define if the training session is executed.
   - do_eval (store_value): boolean value to define if the evaluation session is executed.
   - do_cross_disease_training (store_value): boolean value to define if cross-disease mode is applied.
   - model_id (int): the model id for the model selection, precisely:
     - 1: for the LMCE model
     - 2: for the LMCE_mul model
     - 3: for the LMCE_bilinear model
   - epoch (int): the number of training epochs
   - batch_size (int): the batch size for the training session
   - eval_batch_size (int): the batch size for the evaluation session
   - embed_mode (string): the backbone Language Model, "PubMedBERT_base" and "PubMedBERT_large" are supported.
   - exp_setting (string): the experimental setting for the task (correlation detection): binary or multi_class
   - eval_metric (string): micro or macro evaluation (f1-score)
   - lr (float): the learning rate
   - weight_decay (float): weight decaying rate
   - dropout (float): the dropout rate
   - seed (int): random seed initialization 
   - output_dir (string): the output directory
   - output_file (string): the name of the result file
   - sentence_wise_splits (store_value): whether or not to split the dataset in a sentence-wise way
   - fold (int): the id of the split if cross-validation is applied
   - aggregation (string): the aggregation strategy after the LM (potential values: 'start_start', 'end_end'
    'cls_start_start', 'cls_end_end', 'start_inter_start', 'end_inter_end', 'start_end_start_end', 
    'cls_start_end_start_end', 'start_end_inter_start_end')


