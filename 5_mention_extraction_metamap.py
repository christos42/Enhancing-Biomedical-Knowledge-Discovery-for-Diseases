'''
This software, “MetaMapLite” was developed and funded by the National Library of Medicine, part of the National Institutes of Health,
and agency of the United States Department of Health and Human Services, which is making the software available to the public for any
commercial or non-commercial purpose under the following open-source BSD license.

NOTE: Users of the data distributed with MetaMapLite are responsible for compliance with the UMLS Metathesaurus License Agreement
which requires you to respect the copyrights of the constituent vocabularies and to file a brief annual report on your use of the UMLS.
You also must have activated a UMLS Terminology Services (UTS) account.
'''

import argparse
from pymetamap import MetaMapLite
import pandas as pd
import os
import json

def save_json(file, name, output_path = ''):
    with open(output_path + name, "w") as outfile:
        json.dump(file, outfile)


def read_json(f_path):
    f = open(f_path)
    data = json.load(f)
    return data


def find_json_files(path):
    f_path = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith('.json'):
                f_path.append(os.path.join(root, name))

    return f_path


def create_new_folder(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)


def get_keys_from_mm(concept, klist):
    conc_dict = concept._asdict()
    conc_list = [conc_dict.get(kk) for kk in klist]
    return(tuple(conc_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the date of extraction")
    parser.add_argument("--metamap_path", default="metamap/public_mm_lite/", type=str,
                        required=True, help="the path to metamap installation")
    parser.add_argument("--input_path", default="output/abstracts/", type=str,
                        required=False, help="the path that contains the abstracts")
    parser.add_argument("--output_path", default="output/mentions_extraction/",
                        type=str, required=False, help="the path of the files with extracted mentions/entities")

    args = parser.parse_args()

    output_path = args.output_path + args.date + '/metamap/'
    create_new_folder(output_path)

    files = find_json_files(args.input_path + args.date + '/')

    mm = MetaMapLite.get_instance(args.metamap_path)
    keys_of_interest = ['preferred_name', 'cui', 'score', 'semtypes', 'pos_info', 'trigger']

    # https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
    semantic_types = ['aapp', 'acab', 'amas', 'amph', 'anab', 'anim', 'anst', 'antb', 'arch',
                      'bacs', 'bact', 'bdsu', 'bdsy', 'bhvr', 'biof', 'blor', 'bodm', 'bpoc',
                      'bsoj', 'celc', 'celf', 'cell', 'cgab', 'chem', 'chvf', 'chvs', 'clna',
                      'clnd', 'comd', 'crbs', 'diap', 'dora', 'dsyn', 'eehu', 'elii', 'emod',
                      'emst', 'enzy', 'euka', 'ffas', 'fngs', 'food', 'genf', 'gngm', 'hcpp',
                      'hlca', 'hops', 'horm', 'imft', 'inbe', 'inch', 'inpo', 'irda', 'lbpr',
                      'lbtr', 'mamm', 'mbrt', 'menp', 'mobd', 'mosq', 'neop', 'nnon', 'nusq',
                      'orch', 'orga', 'orgf', 'orgm', 'ortf', 'patf', 'phsu', 'plnt', 'popg',
                      'rcpt', 'rept', 'sbst', 'socb', 'sosy', 'tisu', 'topp', 'virs', 'vita',
                      'vtbt']

    for f in files:
        file_name = f.split('/')[-1]
        e2e_output_path = output_path + file_name.split('.')[0] + '/'
        create_new_folder(e2e_output_path)
        data = read_json(f)
        for k1 in data:
            for i, sent in enumerate(data[k1]['abstract_tokenized']):
                concs, error = mm.extract_concepts([sent],
                                                   restrict_to_sts=semantic_types)

                if len(concs) > 0:
                    cols = [get_keys_from_mm(cc, keys_of_interest) for cc in concs]
                    results_df = pd.DataFrame(cols, columns=keys_of_interest)

                    results_df.to_csv(e2e_output_path + data[k1]['sentence_ids'][i] + '.csv', index=False)