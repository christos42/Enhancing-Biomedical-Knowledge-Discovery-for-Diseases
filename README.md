# 

# Knowledge Extraction for incurable diseases using pubmed: A step towards never-ending learning
Official code repository of "Knowledge Extraction for incurable diseases using pubmed: A step towards never-ending learning" paper.
For a detailed description of the framework, see our paper: 

---

## Setup
### Requirements
 - Python 3.5+
 - BioPython (tested with version 1.78)
 - nltk (tested with version 3.7) 
 - scispacy (tested with version 0.5.1)
 - sklearn (tested with version 0.23.2)
 - spacy (tested with version 3.4.4)

### Execution steps
- Run ```1_extract_pmid_per_disease.py --query [--output_path]``` to extract the related PubMed article PMIDs given the query term (e.g.  <i>python 1_extract_pmid_per_disease.py --query 'rett syndrome'</i>). Arguments:
  - query (string): the query term for the search (e.g. rett syndrome or alzheimer's disease)
  - output_path (string) (optional, default value: <i>output/pmid/</i>): the path to store the extracted PMIDs (paper identifiers provided by PubMed)
- Run ```2_extract_abstracts_per_disease.py --date [--input_path] [--output_path]``` to retrieve the abstracts given the extracted PMIDs of the first step (e.g.  <i>2_extract_abstracts_per_disease.py --date '09_03_23'</i>) [NOTE 1]. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/pmid/</i>): the path where the extracted PMIDs are stored
  - output_path (string) (optional, default value: <i>output/abstracts/</i>): the path to store the extracted abstracts
- Run ```3_abstract_processing.py --date [--input_path]``` to tokenize the sentences of the abstracts. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/abstracts/</i>): the path with the extracted abstracts
- Run ```4_abstract_info.py --date [--input_path]``` to find the number of abstracts per disease and plot the frequency of new released articles per year. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/abstracts/</i>): the path with the extracted abstracts

For the mention extraction two pipelines are provided. One is using the [SciSpacy library](https://github.com/allenai/scispacy) [1] to extract the mentions 
and link them in knowledge bases/vocabularies (e.g. UMLS, MESH, etc.). The other one is using the UMLS MetaMapLite 
tool for mention extraction and mapping.

#### SciSpacy pipeline
The official SciSpacy library utilizes an older UMLS version (2020AA) to train and create the linkers 
(char-3grams string overlap-based search). Follow the next steps to update the linkers:
- [Download](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) the new UMLS version and 
store the files under the folder ```UMLS_update_SciSpacy```.
- Execute the ```knowledge_base_creation.ipynb``` notebook to extract the processed [knowledge bases](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/index.html) 
(UMLS, GO, NCBI, SNOMEDCT_US, HPO, MESH, RXNORM, DRUGBANK, GS).
- Execute the ```tfidf_creation.ipynb``` notebook to create the linkers.
- Find the location of SciSpacy package (i.e. <i>miniconda3/envs/ml_42/lib/python3.8/site-packages/scispacy/</i>). Update the paths of the created linkers accordingly in the scripts ```linking_utils.py``` and ```candidate_generation.py```
that are contained in SciSpacy library. We provide our updated version of the scripts for reference.
- Run ```5_mention_extraction_scispacy.py --date [--linker] [--input_path] [--output_path]``` to extract the entities [NOTE 2] and link them to the defined knowledge base/vocabularies. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - linker (string) (optional, default value: <i>umls</i>): the knowledge base/vocabulary/taxonomy where the extracted entities are linked. Supported linkers: umls, mesh, rxnorm, go, hpo, drugbank, gs, ncbi, snomed 
  - input_path (string) (optional, default value: <i>output/abstracts/</i>): the path with the extracted abstracts
  - output_path (string) (optional, default value: <i>output/mentions_extraction/</i>)
- Run ```6_entity_linking_merge.py --date [--input_path]``` to merge the mapped entities of the different linkers (e.g. UMLS, GO, etc.) [NOTE 3]. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/mentions_extraction/</i>): the path with the extracted mentions
- Run ```7_group_ne_tags.py --date [--input_path]``` to group the extracted entities. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/mentions_extraction/</i>): the path with the extracted mentions
- Run ```8_sampling_linking_codes.py --date [--input_path]``` to sample the mapped/linked entities. Some extracted entities 
  are linked in more than one knowledge bases/vocabularies. Based on their entity type (e.g. disease, gene, etc.) a 
  prioritised sampling strategy is defined and used. For example, for an entity with predicted type <i>chemical/drug</i>,
  we firstly check if there is any linked entity to RXNORM (specific vocabulary for chemicals) in order to sample it.
  Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/mentions_extraction/</i>): the path with the extracted mentions

## Notes
  - [NOTE 1]: Sometimes the <i>efetch</i> API calls in PubMed fail. Consequently, some abstracts might not be successfully retrieved.
            The script prints the name of the PMID files (e.g. 'rett_syndrome.json') that haven't been completed. If all the abstract
            have been successfully extracted the script prints the message: "All abstracts have been extracted successfully!". 
            In the case of API failure, in order to retrieve all the abstracts the script should be executed iteratively 
            until the message that indicates the success is printed. 
  - [NOTE 2]: We utilise the [4 provided models](https://allenai.github.io/scispacy/) (CRAFT, JNLPBA, BC5CDR, BIONLP13CG) to
              extract a wide range of entity types. 
  - [NOTE 3]: In the implementation, we hypothesize that all the supported linkers are used. If this is not the case comment-out
              lines of code accordingly in the ```6_entity_linking_merge.py --date [--input_path]``` script and adjust also the 
              ```merge_linkers_scispacy``` function.

Please cite our work when using this software.

BibTex:
<pre>

</pre>


## References
```
[1] Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019, August). ScispaCy: Fast and Robust Models for 
    Biomedical Natural Language Processing. In Proceedings of the 18th BioNLP Workshop and Shared Task (pp. 319-327).
```