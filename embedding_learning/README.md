# Embedding Learning (LaMEL)
This subdirectory consist of the supervised approaches of the study, presenting the LaMEL model.

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
   --exp_setting --eval_metric --lr --weight_decay --seed --dropout --do_gradient_clipping --clip --steps --margin 
   --threshold --output_dir --output_file --sentence_wise_splits --do_cross_validation --fold --aggregation``` 
   to train and evaluate the models. [NOTE 1]
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
     - 1: for the LaMEL model (E<sub>A</sub>, E<sub>B</sub>, E<sub>C</sub>, E<sub>D</sub>)
     - 2: for the LaMEL_inter model (E<sub>E</sub>, E<sub>F</sub>, E<sub>G</sub>, E<sub>H</sub>)
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
   - margin (float): the margin of the loss function, Should be a number from −1 to 1 (0 to 0.5 is suggested)
   - threshold (float): the threshold for cosine similarity to define the decision boundary (no relation, relation)
   - output_dir (string): the output directory
   - output_file (string): the name of the result file
   - sentence_wise_splits (store_value): boolean value to define if dataset split in a sentence-wise way is applied.
   - do_cross_validation (store_value): boolean value to define if cross-validation is executed.
   - fold (int): the id of the split if cross-validation is applied
   - aggregation (string): the aggregation strategy after the LM ("ent_context_ent_context" --> E<sub>D</sub> (model 1) & E<sub>E</sub> (model 2), 
     "start_start" --> E<sub>A</sub> (model 1) & E<sub>F</sub> (model 2), "end_end" --> E<sub>B</sub> (model 1) & E<sub>G</sub> (model 2), 
     "start_end_start_end" --> E<sub>C</sub> (model 1) & E<sub>H</sub> (model 2))
---

### Notes
  - [NOTE 1] : Example running scripts are provided in <i>run_scripts</i> folder.


