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


def get_chunk(pos):
    start = int(pos.split('/')[0]) - 1
    stop = start + int(pos.split('/')[1])
    return [start, stop]


def merge_sequent_entities(en1, en2, chunk1, chunk2):
    m_ent = {'preferred_name': en1['preferred_name'] + '||' + en2['preferred_name'],
             'cui': en1['cui'] + '||' + en2['cui'],
             'semantic_type': en1['semantic_type'] + en2['semantic_type'],
             'position': str(chunk1[0]) + '/' + str(chunk2[1] - chunk1[0]),
             'score': [en1['score'], en2['score']],
             'trigger': en1['trigger'] + '||' + en2['trigger'],
             'mapped_semantic_type': en1['mapped_semantic_type'] + en2['mapped_semantic_type']}
    return m_ent


def detect_overlaps(positions, d_):
    overlaps = []
    for i1, p1 in enumerate(positions):
        for i2, p2 in enumerate(positions):
            if i1 == i2:
                continue
            else:
                p1_start = int(p1.split('/')[0]) - 1
                p1_stop = p1_start + int(p1.split('/')[1])
                p2_start = int(p2.split('/')[0]) - 1
                p2_stop = p2_start + int(p2.split('/')[1])
                if (p1_start <= p2_start) and (p2_start <= p1_stop):
                    cui1 = d_[p1]['cui']
                    cui2 = d_[p2]['cui']
                    if cui1 == cui2:
                        flag = 0
                        for i3, o in enumerate(overlaps):
                            if i1 in o:
                                flag = 1
                                overlaps[i3].append(i2)
                            elif i2 in o:
                                flag = 1
                                overlaps[i3].insert(o.index(i2), i1)
                        if flag == 0:
                            overlaps.append([i1, i2])
        
    return overlaps

    
def resolve_overlaps(positions, d_, overlaps):
    keys_to_remove = []
    for o in overlaps:
        p1 = positions[o[0]]
        p2 = positions[o[1]]
        score1 = d_[p1]['score']
        score2 = d_[p2]['score']
        if score1 > score2:
            keys_to_remove.append(p2)
        elif score1 < score2:
            keys_to_remove.append(p1)
        else:
            p1_start = int(p1.split('/')[0]) - 1
            p1_stop = p1_start + int(p1.split('/')[1])
            p2_start = int(p2.split('/')[0]) - 1
            p2_stop = p2_start + int(p2.split('/')[1])
            len1 = p1_stop - p1_start
            len2 = p2_stop - p2_start
            if len1 > len2:
                keys_to_remove.append(p2)
            else:
                keys_to_remove.append(p1)

    return keys_to_remove


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
            semantic_types_reformed = []
            for t in types:
                semantic_types_reformed.append(t.replace(' ', ''))
                try:
                    mapped_types.append(mapping[t.replace(' ', '')])
                except:
                    pass
            all_entities[k1][k2]['mapped_semantic_type'] = mapped_types
            all_entities[k1][k2]['semantic_type'] = semantic_types_reformed

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
                    if ent1['cui'] == ent2['cui']:
                        entities_to_merge.append([i1, i2])
                    elif ent1['semantic_type'] != ent2['semantic_type']:
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

    save_json(all_entities, args.disease + '.json', args.input_path + args.date + '/metamap/merged_entities/')
