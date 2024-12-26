# 

# Enhancing Biomedical Knowledge Discovery for Diseases
Official code repository of "Enhancing Biomedical Knowledge Discovery for Diseases: An Open-Source Framework Applied on Rett Syndrome and Alzheimer’s Disease" publication.
For a detailed description of the framework, see our paper:

Theodoropoulos, C., Catalin Coman, A., Henderson, J., and Moens, M.-F. Enhancing biomedical knowledge discovery for diseases: An open-source framework applied on Rett syndrome and Alzheimer’s disease. IEEE Access 12 (2024), 180652–180673. doi: https://doi.org/10.1109/ACCESS.2024.3509714

---

## Setup
### Requirements
 - Python 3.5+
 - BioPython (tested with version 1.78)
 - nltk (tested with version 3.7) 
 - scispacy (tested with version 0.5.1)
 - sklearn (tested with version 0.23.2)
 - spacy (tested with version 3.4.4)
 - pandas (tested with version 1.4.1)
 - numpy (tested with version 1.23.5)
 - [pymetamap](https://github.com/AnthonyMRios/pymetamap)

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
and link them in knowledge bases/vocabularies (e.g. UMLS [2], MESH, etc.). The other one is using the UMLS MetaMapLite [3] 
tool for mention extraction and mapping.

#### SciSpacy pipeline
The official SciSpacy library utilizes an older UMLS [2] version (2020AA) to train and create the linkers 
(char-3grams string overlap-based search). Follow the next steps to update the linkers:
- [Download](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) the new UMLS version [2] and 
store the files under the folder ```UMLS_update_SciSpacy```.
- Execute the ```knowledge_base_creation.ipynb``` notebook to extract the processed [knowledge bases](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/index.html) 
(UMLS [2], GO [4], NCBI [5], SNOMEDCT_US [6], HPO [7], MESH [8], RXNORM [9], DRUGBANK [10], [GS](https://www.nlm.nih.gov/research/umls/sourcereleasedocs/current/GS/index.html)).
- Execute the ```tfidf_creation.ipynb``` notebook to create the linkers.
- Find the location of SciSpacy package (i.e. <i>miniconda3/envs/ml_42/lib/python3.8/site-packages/scispacy/</i>). Update the paths of the created linkers accordingly in the scripts ```linking_utils.py``` and ```candidate_generation.py```
that are contained in SciSpacy library. We provide our updated version of the scripts for reference.
- Run ```5_mention_extraction_scispacy.py --date [--linker] [--input_path] [--output_path]``` to extract the entities [NOTE 2] and link them to the defined knowledge base/vocabularies. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - linker (string) (optional, default value: <i>umls</i>): the knowledge base/vocabulary/taxonomy where the extracted entities are linked. Supported linkers: umls, mesh, rxnorm, go, hpo, drugbank, gs, ncbi, snomed 
  - input_path (string) (optional, default value: <i>output/abstracts/</i>): the path with the extracted abstracts
  - output_path (string) (optional, default value: <i>output/mentions_extraction/</i>): : the path with the extracted mentions
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
- Run ```9_cooccurrence_extraction.py --pipeline --date [--input_path]``` to extract the co-occurrence graph with the frequencies.
  Arguments:
  - pipeline (string): supported pipelines (scispacy or metamap) 
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/mentions_extraction/</i>): the path with the extracted mentions

#### Metamap pipeline
- Install [MetaMap Lite](https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/run-locally/MetaMapLite.html) locally. 
- Move the script ```5_mention_extraction_metamap.py``` under the folder where MetaMap (e.g. <i>metamap/public_mm_lite</i>) is installed.
- Run ```5_mention_extraction_metamap.py --date --metamap_path [--input_path] [--output_path]``` to extract the mentions using
  the MetaMap Lite tool. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - metamap_path (string) (default value: <i>metamap/public_mm_lite/</i>): the path to metamap installation
  - input_path (string) (optional, default value: <i>output/abstracts/</i>): the path with the extracted abstracts
  - output_path (string) (optional, default value: <i>output/mentions_extraction/</i>): : the path with the extracted mentions
- Run ```6_mention_merge_metamap.py --date --disease --entity_expansion [--input_path] [--abstract_path]``` to process and merge the extracted mentions. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - disease (string) (e.g. <i>rett_syndrome</i>): the name of the disease
  - entity_expansion (int): whether the entity expansion strategy is applied or not [NOTE 4]
  - input_path (string) (optional, default value: <i>output/mentions_extraction/</i>): : the path with the extracted mentions
  - abstract_path (string) (optional, default value: <i>output/abstracts/</i>): the path with the extracted abstracts
- Run ```9_cooccurrence_extraction.py --pipeline --date [--input_path]``` to extract the co-occurrence graph with the frequencies.
  Arguments:
  - pipeline (string): supported pipelines (scispacy or metamap) 
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/mentions_extraction/</i>): the path with the extracted mentions
---

## License
© 2024. The code is distributed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Notes
  - [NOTE 1] : Sometimes the <i>efetch</i> API calls in PubMed fail. Consequently, some abstracts might not be successfully retrieved.
               The script prints the name of the PMID files (e.g. 'rett_syndrome.json') that haven't been completed. If all the abstract
               have been successfully extracted the script prints the message: "All abstracts have been extracted successfully!". 
               In the case of API failure, in order to retrieve all the abstracts the script should be executed iteratively 
               until the message that indicates the success is printed. 
  - [NOTE 2] : We utilise the [4 provided models](https://allenai.github.io/scispacy/) (CRAFT [11], JNLPBA [12], BC5CDR [13], BIONLP13CG [14]) to
               extract a wide range of entity types. 
  - [NOTE 3] : In the implementation, we hypothesize that all the supported linkers are used. If this is not the case comment-out
               lines of code accordingly in the ```6_entity_linking_merge.py --date [--input_path]``` script and adjust also the 
               ```merge_linkers_scispacy``` function.
  - [NOTE 4] : When the next character of an entity is not one of the following: " ", <, >, (, ), . (in the end of the sentence), then
               expand the entity as far as the aforementioned restriction holds. For example, if a medication with the name <i>drug</i>
               is detected and it is <i>drug4.2%</i> in the context, then the <i>drug</i> entity is expanded to <i>drug4.2%</i>.

Please cite our work when using this software.

BibTex:
<pre>
@article{theodoropoulos2024enhancing,
  author={Theodoropoulos, Christos and Catalin Coman, Andrei and Henderson, James and Moens, Marie-Francine},
  journal={IEEE Access}, 
  title={Enhancing Biomedical Knowledge Discovery for Diseases: An Open-Source Framework Applied on Rett Syndrome and Alzheimer’s Disease}, 
  year={2024},
  volume={12},
  publisher={IEEE},
  pages={180652-180673},
  doi={10.1109/ACCESS.2024.3509714}
}
</pre>




## References
```
[1] Neumann, M., King, D., Beltagy, I., & Ammar, W. (2019, August). ScispaCy: Fast and Robust Models for 
    Biomedical Natural Language Processing. In Proceedings of the 18th BioNLP Workshop and Shared Task (pp. 319-327).
[2] Bodenreider O. The Unified Medical Language System (UMLS): integrating biomedical terminology. 
    Nucleic Acids Res. 2004 Jan 1;32(Database issue):D267-70. doi: 10.1093/nar/gkh061. 
    PubMed PMID: 14681409; PubMed Central PMCID: PMC308795.
[3] Alan R Aronson. 2001. Effective mapping of biomedical text to the umls metathesaurus: the metamap program. 
    In Proceedings of the AMIA Symposium, page 17. American Medical Informatics Association.
[4] Gene Ontology Consortium. 2004. The gene ontology (go) database and informatics resource. 
    Nucleic acids research, 32(suppl_1):D258–D261. 
[5] Conrad L Schoch, Stacy Ciufo, Mikhail Domrachev, Carol L Hotton, Sivakumar Kannan, Rogneda Khovanskaya, Detlef Leipe, 
    Richard Mcveigh, Kathleen O’Neill, Barbara Robbertse, et al. 2020. Ncbi taxonomy: a comprehensive update on curation, 
    resources and tools. Database, 2020:baaa062.
[6] Michael Q Stearns, Colin Price, Kent A Spackman, and Amy Y Wang. 2001. 
    Snomed clinical terms: overview of the development process and project status. 
    In Proceedings of the AMIA Symposium, page 1002. American Medical Informatics Association.
[7] Sebastian Köhler, Michael Gargano, Nicolas Matentzoglu, Leigh C Carmody, David Lewis-Smith, Nicole A Vasilevsky, 
    Daniel Danis, Ganna Balagura, Gareth Baynam, Amy M Brower, et al. 2021. 
    The human phenotype ontology in 2021. Nucleic acids research, 49(D1):D1207–D1217. 
[8] Carolyn E Lipscomb. 2000. Medical subject headings (mesh). Bulletin of the Medical Library Association, 88(3):265.
[9] Stuart J Nelson, Kelly Zeng, John Kilbourne, Tammy Powell, and Robin Moore. 2011. 
    Normalized names for clinical drugs: Rxnorm at 6 years. 
    Journal of the American Medical Informatics Association, 18(4):441–448.
[10] Craig Knox, Mike Wilson, Christen M Klinger, Mark Franklin, Eponine Oler, Alex Wilson, Allison Pon, Jordan Cox, 
    Na Eun Chin, Seth A Strawbridge, et al. 2024. Drugbank 6.0: the drugbank knowledgebase for 2024. 
    Nucleic Acids Research, 52(D1):D1265–D1275. 
[11] Michael Bada, Miriam Eckert, Donald Evans, Kristin Garcia, Krista Shipley, Dmitry Sitnikov, William A Baumgartner, 
     K Bretonnel Cohen, Karin Verspoor, Judith A Blake, et al. 2012. Concept annotation in the craft corpus. 
     BMC bioinformatics, 13:1–20.
[12] Nigel Collier, Tomoko Ohta, Yoshimasa Tsuruoka, Yuka Tateisi, and Jin-Dong Kim. 2004. 
     Introduction to the bio-entity recognition task at JNLPBA. 
     In Proceedings of the International Joint Workshop on Natural Language Processing in Biomedicine 
     and its Applications (NLPBA/BioNLP), pages 73–78, Geneva, Switzerland. COLING. 
[13] Jiao Li, Yueping Sun, Robin J Johnson, Daniela Sciaky, Chih-Hsuan Wei, Robert Leaman, Allan Peter Davis, 
     Carolyn J Mattingly, Thomas C Wiegers, and Zhiyong Lu. 2016. 
     Biocreative v cdr task corpus: a resource for chemical disease relation extraction. Database, 2016.
[14] Jin-Dong Kim, Yue Wang, and Yamamoto Yasunori. 2013. The Genia event extraction shared task, 2013 edition - overview. 
     In Proceedings of the BioNLP Shared Task 2013 Workshop, pages 8–15, Sofia, Bulgaria.
     Association for Computational Linguistics.
```