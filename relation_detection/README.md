# Relation Detection (LaMReDA & LaMReDM)
This subdirectory consist of the supervised approaches of the study, presenting the LaMReDA and LaMReDM models and the probing experiments.

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
 - Run ```main.py [--dataset_path] [--dataset_path_eval] [--dataset_path_train] [--dataset_path_dev] [--dataset_path_test] 
   --do_train --do_eval --do_cross_disease_training --model_id --epoch --batch_size --eval_batch_size --embed_mode 
   --exp_setting --eval_metric --lr --weight_decay --seed --dropout --do_gradient_clipping --clip --steps --output_dir
   --output_file --sentence_wise_splits --do_cross_validation --fold --aggregation --save_best_weights 
   --use_distantly_supervised_data --projection_dimension``` to train and evaluate the models. 
   Arguments:
   - dataset_path (string): the path of the dataset
   - dataset_path_eval (string): the path of the evaluation dataset in case that the training and evaluation 
     datasets are different files. The argument is used when cross-disease training and evaluation is applied.
   - dataset_path_train (string): the path of the training dataset (original split setup)
   - dataset_path_dev (string): the path of the development dataset (original split setup)
   - dataset_path_train (string): the path of the test dataset (original split setup)
   - do_train (store_value): boolean value to define if the training session is executed.
   - do_eval (store_value): boolean value to define if the evaluation session is executed.
   - do_cross_disease_training (store_value): boolean value to define if cross-disease mode is applied.
   - model_id (int): the model id for the model selection, precisely:
     - 1: for the LaMReDA model
     - 2: for the LaMReDM model
   - epoch (int): the number of training epochs
   - batch_size (int): the batch size for the training session
   - eval_batch_size (int): the batch size for the evaluation session
   - embed_mode (string): the backbone Language Model, "PubMedBERT_base" and "PubMedBERT_large" are supported.
   - exp_setting (string): the experimental setting for the task (correlation detection): binary or multi_class
   - eval_metric (string): micro or macro evaluation (f1-score)
   - lr (float): the learning rate
   - weight_decay (float): weight decaying rate
   - seed (int): random seed initialization 
   - dropout (float): the dropout rate
   - do_gradient_clipping (store_value): boolean value to define if gradient clipping is applied.
   - clip (float): the max norm of the gradient when gradient clipping is applied
   - steps (int): number of training steps for result monitoring
   - output_dir (string): the output directory
   - output_file (string): the name of the result file
   - sentence_wise_splits (store_value): boolean value to define if dataset split in a sentence-wise way is applied.
   - do_cross_validation (store_value): boolean value to define if cross-validation is executed.
   - fold (int): the id of the split if cross-validation is applied
   - aggregation (string): the aggregation strategy after the LM ("start_start" --> R<sub>A</sub>, 
     "end_end" --> R<sub>B</sub>, "ent_context_ent_context" --> R<sub>D</sub> , "start_end_start_end" --> R<sub>C</sub>,
     "inter" --> R<sub>E</sub>, "cls_start_start" --> R<sub>G</sub>, "cls_end_end" --> R<sub>H</sub>, 
     "cls_ent_context_ent_context" --> R<sub>F</sub>, "cls_inter" --> R<sub>J</sub>, 
     "cls_start_end_start_end" --> R<sub>I</sub>, "start_inter_start" --> R<sub>K</sub>, 
     "end_inter_end" --> R<sub>L</sub>, "start_end_inter_start_end" --> R<sub>M</sub>,
     "ent_context_inter_ent_context" --> R<sub>N</sub>, "atlop_context_vector_only" --> R<sub>O</sub>, 
     "atlop_context_vector" --> R<sub>P</sub>)
   - save_best_weights (store_value): boolean value to define if the weights based on the best performance on the 
     development set will be saved.
   - use_distantly_supervised_data (store_value): boolean value to define if the distantly supervised data will be used
     (weakly supervised setup)
   - projection_dimension (int): the dimension of the projection layer (paper details: 768 and 1024 when PubMedBERT base 
     and PubMedBERT large is used, respectively.), the default value is 0 and indicates that the projection dimension is 
     equal to the embedding size.

---

### Notes
  - [NOTE 1] : Example running scripts are provided in <i>run_scripts</i> folder.


