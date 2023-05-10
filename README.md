# 

# Knowledge Extraction for incurable diseases using pubmed: A step towards never-ending learning
Official code repository of "Knowledge Extraction for incurable diseases using pubmed: A step towards never-ending learning" paper.
For a detailed description of the framework, see our paper: 

## Setup
### Requirements
 - Python 3.5+
 - BioPython (tested with version 1.78)
 - nltk (tested with version 3.7) 

### Execution steps
- Run the ```1_extract_pmid_per_disease.py --query [--output_path]``` script to extract the related PubMed article PMIDs given the query term (e.g.  <i>python 1_extract_pmid_per_disease.py --query 'rett syndrome'</i>). Arguments:
  - query (string): the query term for the search (e.g. rett syndrome or alzheimer's disease)
  - output_path (string) (optional, default value: <i>output/pmid/</i>): the path to store the extracted PMIDs (paper identifiers provided by PubMed)
- Run the ```2_extract_abstracts_per_disease.py --date [--input_path] [--output_path]``` to retrieve the abstracts given the extracted PMIDs of the first step (e.g.  <i>2_extract_abstracts_per_disease.py --date '09_03_23'</i>) [NOTE 1]. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/pmid/</i>): the path where the extracted PMIDs are stored
  - output_path (string) (optional, default value: <i>output/abstracts/</i>): the path to store the extracted abstracts
- Run the ```3_abstract_processing.py --date [--input_path]``` to tokenize the sentences of the abstracts. Arguments:
  - date (string): the date of the PMID extraction in the following format: day_month_year
  - input_path (string) (optional, default value: <i>output/abstracts/</i>): the path with the extracted abstracts
- 

## Notes
  [NOTE 1]: Sometimes the <i>efetch</i> API calls in PubMed fail. Consequently, some abstracts might not be successfully retrieved.
            The script prints the name of the PMID files (e.g. 'rett_syndrome.json') that haven't been completed. If all the abstract
            have been successfully extracted the script prints the message: "All abstracts have been extracted successfully!". 
            In the case of API failure, in order to retrieve all the abstracts the script should be executed iteratively 
            until the message that indicates the success is printed. 

Please cite our work when using this software.

BibTex:
<pre>

</pre>


## References
```
[1] 
```