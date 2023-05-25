import argparse
import pandas as pd
import os
from utils.utils import find_csv_files, save_json, read_json
from utils.metamap_concepts_utils import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the date of extraction")
    parser.add_argument("--disease", default="rett_syndrome", type=str,
                        required=True, help="the disease name written with underscores")
    parser.add_argument("--entity_expansion", default=1, type=int, 
                        required=True, help="whether the entity expansion is applied or not")
    parser.add_argument("--input_path", default="output/mentions_extraction/",
                        type=str, required=False, help="the path of the files with extracted mentions/entities")
    parser.add_argument("--abstract_path", default="output/abstracts/", type=str, 
                        required=False, help="the path to the abstract file")

    args = parser.parse_args()

    abstracts = read_json(args.abstract_path + args.date + '/' + args.disease + '.json')

    mapping = {'aapp': 'Amino Acid, Peptide, or Protein',
               'acab': 'Acquired Abnormality',
               'amas': 'Amino Acid Sequence',
               'amph': 'Amphibian',
               'anab': 'Anatomical Abnormality',
               'anim': 'Animal',
               'anst': 'Anatomical Structure',
               'antb': 'Antibiotic',
               'arch': 'Archaeon',
               'bacs': 'Biologically Active Substance',
               'bact': 'Bacterium',
               'bdsu': 'Body Substance',
               'bdsy': 'Body System',
               'bhvr': 'Behavior',
               'biof': 'Biologic Function',
               'blor': 'Body Location or Region',
               'bodm': 'Biomedical or Dental Material',
               'bpoc': 'Body Part, Organ, or Organ Component',
               'bsoj': 'Body Space or Junction',
               'celc': 'Cell Component',
               'celf': 'Cell Function',
               'cell': 'Cell',
               'cgab': 'Congenital Abnormality',
               'chem': 'Chemical',
               'chvf': 'Chemical Viewed Functionally',
               'chvs': 'Chemical Viewed Structurally',
               'clna': 'Clinical Attribute',
               'clnd': 'Clinical Drug',
               'comd': 'Cell or Molecular Dysfunction',
               'crbs': 'Carbohydrate Sequence',
               'diap': 'Diagnostic Procedure',
               'dora': 'Daily or Recreational Activity',
               'dsyn': 'Disease or Syndrome',
               'eehu': 'Environmental Effect of Humans',
               'elii': 'Element, Ion, or Isotope',
               'emod': 'Experimental Model of Disease',
               'emst': 'Embryonic Structure',
               'enzy': 'Enzyme',
               'euka': 'Eukaryote',
               'ffas': 'Fully Formed Anatomical Structure',
               'fngs': 'Fungus',
               'food': 'Food',
               'genf': 'Genetic Function',
               'gngm': 'Gene or Genome',
               'hcpp': 'Human-caused Phenomenon or Process',
               'hlca': 'Health Care Activity',
               'hops': 'Hazardous or Poisonous Substance',
               'horm': 'Hormone',
               'imft': 'Immunologic Factor',
               'inbe': 'Individual Behavior',
               'inch': 'Inorganic Chemical',
               'inpo': 'Injury or Poisoning',
               'irda': 'Indicator, Reagent, or Diagnostic Aid',
               'lbpr': 'Laboratory Procedure',
               'lbtr': 'Laboratory or Test Result',
               'mamm': 'Mammal',
               'mbrt': 'Molecular Biology Research Technique',
               'menp': 'Mental Process',
               'mobd': 'Mental or Behavioral Dysfunction',
               'mosq': 'Molecular Sequence',
               'neop': 'Neoplastic Process',
               'nnon': 'Nucleic Acid, Nucleoside, or Nucleotide',
               'nusq': 'Nucleotide Sequence',
               'orch': 'Organic Chemical',
               'orga': 'Organism Attribute',
               'orgf': 'Organism Function',
               'orgm': 'Organism',
               'ortf': 'Organ or Tissue Function',
               'patf': 'Pathologic Function',
               'phsu': 'Pharmacologic Substance',
               'plnt': 'Plant',
               'popg': 'Population Group',
               'rcpt': 'Receptor',
               'rept': 'Reptile',
               'sbst': 'Substance',
               'socb': 'Social Behavior',
               'sosy': 'Sign or Symptom',
               'tisu': 'Tissue',
               'topp': 'Therapeutic or Preventive Procedure',
               'virs': 'Virus',
               'vita': 'Vitamin',
               'vtbt': 'Vertebrate'}

    files = find_csv_files(args.input_path + args.date + '/metamap/' + args.disease + '/')
    all_entities = {}
    for f in files:
        try:
            abstract_id = f.split('/')[-1].split('.')[0]
            d = pd.read_csv(f)
            entities = get_entities(d)
            all_entities[abstract_id] = entities
        except:
            pass

    for k1 in all_entities:
        for k2 in all_entities[k1]:
            mapped_types = []
            types = all_entities[k1][k2]['semantic_type'][1:-1].split(',')
            semantic_types_reformed = []
            for t in types:
                semantic_types_reformed.append(t.replace(' ', ''))
                try:
                    mapped_types.append(mapping[t.replace(' ', '')])
                except:
                    pass
            all_entities[k1][k2]['mapped_semantic_type'] = mapped_types
            all_entities[k1][k2]['semantic_type'] = semantic_types_reformed

    if args.entity_expansion == 1:
        for k in all_entities:
            # Expand the detected entities if possible.
            abstr_k1 = k.split('_')[0]
            abstr_in = int(k.split('_')[1])
            all_entities[k] = expand_entities(all_entities[k], abstracts[abstr_k1]['abstract_tokenized'][abstr_in - 1])

    for k in all_entities:
        chunks = []
        entities_to_merge = []
        for ent in all_entities[k]:
            chunks.append(get_chunk(all_entities[k][ent]['position']))
        # Find the entities that should be merged
        for i1, c1 in enumerate(chunks):
            for i2, c2 in enumerate(chunks):
                if c1[1] + 1 == c2[0]:
                    key1 = list(all_entities[k].keys())[i1]
                    key2 = list(all_entities[k].keys())[i2]
                    ent1 = all_entities[k][key1]
                    ent2 = all_entities[k][key2]
                    #if ent1['cui'] == ent2['cui']:
                    #    entities_to_merge.append([i1, i2])
                    #elif ent1['semantic_type'] != ent2['semantic_type']:
                    #    entities_to_merge.append([i1, i2])
                    entities_to_merge.append([i1, i2])

        keys_to_remove = []
        for m_ent in entities_to_merge:
            key1 = list(all_entities[k].keys())[m_ent[0]]
            key2 = list(all_entities[k].keys())[m_ent[1]]
            ent1 = all_entities[k][key1]
            ent2 = all_entities[k][key2]
            chunk1 = chunks[m_ent[0]]
            chunk2 = chunks[m_ent[1]]
            # Merge the entities
            f_m_ent = merge_sequent_entities(ent1, ent2, chunk1, chunk2)
            # Remove the entities to add the merged version
            keys_to_remove.append(key1)
            keys_to_remove.append(key2)
            all_entities[k][f_m_ent['position']] = f_m_ent
        for k_r in keys_to_remove:
            try:
                all_entities[k].pop(k_r)
            except:
                pass

        # Deal with overlaps
        overlaps = detect_overlaps(list(all_entities[k].keys()), all_entities[k])
        keys_to_remove = resolve_overlaps(list(all_entities[k].keys()), all_entities[k], overlaps)
        for k_r in keys_to_remove:
            all_entities[k].pop(k_r)

    # Deal with overlaps using expansion
    for k in all_entities:
        keys_to_remove, merged_entities = resolve_overlaps_with_expansion(list(all_entities[k].keys()), all_entities[k].copy())
        for m_ent in merged_entities:
            all_entities[k][m_ent['position']] = m_ent
        for k_r in keys_to_remove:
            try:
                all_entities[k].pop(k_r[0])
            except:
                pass
            try:
                all_entities[k].pop(k_r[1])
            except:
                pass


    save_json(all_entities, args.disease + '.json', args.input_path + args.date + '/metamap/merged_entities/')
