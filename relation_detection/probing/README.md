# Probing Experiments
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
 - Run ```main.py --dataset_path --do_train --do_eval --model_id --epoch --batch_size --eval_batch_size --embed_mode 
   --exp_setting --eval_metric --lr --weight_decay --seed --dropout --do_gradient_clipping --clip --steps --output_dir
   --output_file --sentence_wise_splits --fold --aggregation --encoding_layer --attention_head``` 
   to train and evaluate the models. [NOTE 1]
   Arguments:
   - dataset_path (string): the path of the dataset
   - do_train (store_value): boolean value to define if the training session is executed.
   - do_eval (store_value): boolean value to define if the evaluation session is executed.
   - model_id (int): the model id for the model selection, precisely:
     - 1: for the LMREA model (aggregation: addition)
     - 2: for the LMREA_proj model (aggregation: addition) with linear projection layer before the classification layer
     - 3: for the LMREM model (aggregation: multiplication)
     - 4: for the LMREM_proj model (aggregation: multiplication) with linear projection layer before the classification layer
     - 5: for the LMRE_attention model (attention scores)
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
   - fold (int): the id of the split if cross-validation is applied
   - aggregation (string): the aggregation strategy after the LM ("ent_context_ent_context" --> R<sub>D</sub> , 
     "atlop_context_vector_only" --> R<sub>O</sub>, "atlop_context_vector" --> R<sub>P</sub>, 
     "layer_specific" --> attention scores, "head_specific" --> attention scores, 'non_specific' --> attention scores (all)
   - encoding_layer (int): the encoding layer of the LM to extract the representation
   - attention_head (int): the attention head of the LM to extract the attention scores
---

### Notes
  - [NOTE 1] : Example running scripts are provided in <i>run_scripts</i> folder.


