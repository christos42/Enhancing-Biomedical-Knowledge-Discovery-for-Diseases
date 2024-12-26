## Annotation application 
---
We describe the process of sentence sampling and annotation. In this implementation, the output of the Metamap pipeline is used.

## Setup
### Requirements
 - numpy (tested with version 1.23.5)
 - streamlit (tested with version 1.20.0)
 - scispacy (tested with version 0.5.1)

### Execution steps
- Run ```1_sampling_strategy.py --entity_path --cooccurrence_path --abstract_path --disease_name --strategy_id [--n] [--n_conc]``` to sample sentences based on the sampling strategy. Arguments:
  - entity_path (string): the path to the merged extracted entities
  - cooccurrence_path (string): the path to the co-occurrence graph
  - abstract_path (string): the path to the abstract file
  - disease_name (string): the name of the disease (i.e. <i>rett_syndrome</i> or <i>alzheimer_s_disease</i>)
  - strategy_id (string): the id of the sentence sampling strategy (supported values: 1 and 2). The first strategy is the following: appr. 50% of sentences from pool 1 (higher frequency concept pairs, higher probability) + appr. 50% of sentences from pool 2 (lower frequency concept pairs, higher probability). The second strategy has the objective to cover as many concept pairs as possible.
  - n (int) (optional, default value: <i>25</i>): the number of sampled sentences per bucket (sampling strategy: 1)
  - n_conc (int) (optional, default value: <i>10</i>): the number of concept pairs to be sampled per bucket (sampling strategy: 2)
- Run ```2_prepare_sampled_sentences.py --entity_path --abstract_path --disease_name --strategy_id --bucket_id``` to prepare the sentence for the annotation app. Arguments:
  - entity_path (string): the path to the merged extracted entities
  - abstract_path (string): the path to the abstract file
  - disease_name (string): the name of the disease (i.e. <i>rett_syndrome</i> or <i>alzheimer_s_disease</i>)
  - strategy_id (string): the id of the sentence sampling strategy (supported values: 1 and 2)
  - bucket_id (int): the id (number) of the bucket with sentences for annotation
- Run ```streamlit run annotation_app_s.py -- --bucket_id --trial_id --disease_name --annotator``` to start the annotation application. Arguments:
  - bucket_id (int): : the id (number) of the bucket with sentences for annotation
  - trial_id (int): : the id (number) of the execution trial (useful for file naming and progress tracking)
  - disease_name (string): the name of the disease (i.e. <i>rett_syndrome</i> or <i>alzheimer_s_disease</i>)
  - annotator (string): the name of the annotator/expert
- Run ```python 3_dataset_creation.py --entity_path --abstract_path --disease_name --annotator``` to process the annotations and create the datasets. Arguments:
  - entity_path (string): the path to the merged extracted entities
  - abstract_path (string): the path to the abstract file
  - disease_name (string): the name of the disease (i.e. <i>rett_syndrome</i> or <i>alzheimer_s_disease</i>)
  - annotator (string): the name of the annotator/expert
- Run ```python 4_dataset_processing.py --annotator --disease_name``` to process the dataset; tokenize the sentences, add the special tokens, update the entity spans.
  Arguments:
  - annotator (string): the name of the annotator/expert
  - disease_name (string): the name of the disease (i.e. <i>rett_syndrome</i> or <i>alzheimer_s_disease</i>)

#### Datasets
- [NOTE 1] : The ReDReS and ReDAD datasets are available [here](https://drive.google.com/file/d/1PRVORwFszsPfCylzYd0XY1JKZLRgLCtV/view?usp=sharing).
