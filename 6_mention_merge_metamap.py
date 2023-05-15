import argparse
import pandas as pd
import os
from utils.utils import find_csv_files, save_json

def get_entities(d):
    entities = {}
    for r in d.itertuples():
        if r.score < 0.9:
            continue
        pos = r.pos_info.split(';')
        for p in pos:
            if p not in entities:
                entities[p] = {'preferred_name': r.preferred_name,
                               'cui': r.cui,
                               'semantic_type': r.semtypes,
                               'position': p,
                               'score': r.score,
                               'trigger': r.trigger}

    return entities

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the date of extraction")
    parser.add_argument("--disease", default="rett_syndrome", type=str,
                        required=True, help="the disease name written with underscores")
    parser.add_argument("--input_path", default="output/mentions_extraction/",
                        type=str, required=False, help="the path of the files with extracted mentions/entities")

    args = parser.parse_args()

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
            for t in types:
                try:
                    mapped_types.append(mapping[t.replace(' ', '')])
                except:
                    pass
            all_entities[k1][k2]['mapped_semantic_type'] = mapped_types

    save_json(all_entities, args.disease + '.json', args.input_path + args.date + '/metamap/merged_entities/')
