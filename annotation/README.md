## Annotation application 
---
We describe the process of sentence sampling and annotation.

## Setup
### Requirements
 - numpy (tested with version 1.23.5)
 - 

### Execution steps
- Run ```1_sampling_strategy.py --entity_path --cooccurrence_path --abstract_path --disease_name --strategy_id [--n] [--n_conc]``` to sample sentences based on the sampling strategy. Arguments:
  - entity_path (string): the path to the merged extracted entities
  - cooccurrence_path (string): the path to the co-occurrence graph
  - abstract_path (string): the path to the abstract file
  - disease_name (string): the name of the disease (i.e. <i>rett_syndrome</i> or <i>alzheimer_s_disease</i>)
  - strategy_id (string): the id of the sentence sampling strategy (supported values: 1 and 2). The first strategy is the following: appr. 50% of sentences from pool 1 (higher frequency concept pairs, higher probability) + appr. 50% of sentences from pool 2 (lower frequency concept pairs, higher probability). The second strategy has the objective to cover as many concept pairs as possible.
  - n (int) (optional, default value: <i>25</i>): the number of sampled sentences per bucket (sampling strategy: 1)
  - n_conc (int) (optional, default value: <i>10</i>): the number of concept pairs to be sampled per bucket (sampling strategy: 2)
  