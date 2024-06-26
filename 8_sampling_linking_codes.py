import argparse
from utils.utils import find_json_files, read_json, save_json, create_new_folder


def sampling_linking_codes_strategy(data_merged):
    data_merged_upd = data_merged.copy()
    for k1 in data_merged_upd:
        for k2 in data_merged_upd[k1]:
            sampled_linked_ent_list = []
            for i, ent in enumerate(data_merged_upd[k1][k2]['entities']):
                sampled_linked_ent = []
                linked_ent = data_merged_upd[k1][k2]['linked_entities'][i]
                sampled_linked_ent_sub = {'cui': [],
                                          'name': [],
                                          'alias': [],
                                          'tui': [],
                                          'description': [],
                                          'probability': [],
                                          'linker': []}
                for tag in list(set(ent['grouped_type'])):
                    if tag == 'CHEMICAL':
                        if len(linked_ent['rxnorm']['cui']) > 0:
                            dict_to_add = linked_ent['rxnorm'].copy()
                            dict_to_add['linker'] = 'rxnorm'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['gs']['cui']) > 0:
                            dict_to_add = linked_ent['gs'].copy()
                            dict_to_add['linker'] = 'gs'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['drugbank']['cui']) > 0:
                            dict_to_add = linked_ent['drugbank'].copy()
                            dict_to_add['linker'] = 'drugbank'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['mesh']['cui']) > 0:
                            dict_to_add = linked_ent['mesh'].copy()
                            dict_to_add['linker'] = 'mesh'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['hpo']['cui']) > 0:
                            dict_to_add = linked_ent['hpo'].copy()
                            dict_to_add['linker'] = 'hpo'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['go']['cui']) > 0:
                            dict_to_add = linked_ent['go'].copy()
                            dict_to_add['linker'] = 'go'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['ncbi']['cui']) > 0:
                            dict_to_add = linked_ent['ncbi'].copy()
                            dict_to_add['linker'] = 'ncbi'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['snomed']['cui']) > 0:
                            dict_to_add = linked_ent['snomed'].copy()
                            dict_to_add['linker'] = 'snomed'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['umls']['cui']) > 0:
                            dict_to_add = linked_ent['umls'].copy()
                            dict_to_add['linker'] = 'umls'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        else:
                            sampled_linked_ent.append({})
                    elif tag in ['GENE_OR_PROTEIN', 'DNA', 'RNA', 'SO']:
                        if len(linked_ent['go']['cui']) > 0:
                            dict_to_add = linked_ent['go'].copy()
                            dict_to_add['linker'] = 'go'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['hpo']['cui']) > 0:
                            dict_to_add = linked_ent['hpo'].copy()
                            dict_to_add['linker'] = 'hpo'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['rxnorm']['cui']) > 0:
                            dict_to_add = linked_ent['rxnorm'].copy()
                            dict_to_add['linker'] = 'rxnorm'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['gs']['cui']) > 0:
                            dict_to_add = linked_ent['gs'].copy()
                            dict_to_add['linker'] = 'gs'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['drugbank']['cui']) > 0:
                            dict_to_add = linked_ent['drugbank'].copy()
                            dict_to_add['linker'] = 'drugbank'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['mesh']['cui']) > 0:
                            dict_to_add = linked_ent['mesh'].copy()
                            dict_to_add['linker'] = 'mesh'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['ncbi']['cui']) > 0:
                            dict_to_add = linked_ent['ncbi'].copy()
                            dict_to_add['linker'] = 'ncbi'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['snomed']['cui']) > 0:
                            dict_to_add = linked_ent['snomed'].copy()
                            dict_to_add['linker'] = 'snomed'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['umls']['cui']) > 0:
                            dict_to_add = linked_ent['umls'].copy()
                            dict_to_add['linker'] = 'umls'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        else:
                            sampled_linked_ent.append({})
                    elif tag in ['DISEASE', 'PATHOLOGICAL_FORMATION']:
                        if len(linked_ent['hpo']['cui']) > 0:
                            dict_to_add = linked_ent['hpo'].copy()
                            dict_to_add['linker'] = 'hpo'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['mesh']['cui']) > 0:
                            dict_to_add = linked_ent['mesh'].copy()
                            dict_to_add['linker'] = 'mesh'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['rxnorm']['cui']) > 0:
                            dict_to_add = linked_ent['rxnorm'].copy()
                            dict_to_add['linker'] = 'rxnorm'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['gs']['cui']) > 0:
                            dict_to_add = linked_ent['gs'].copy()
                            dict_to_add['linker'] = 'gs'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['drugbank']['cui']) > 0:
                            dict_to_add = linked_ent['drugbank'].copy()
                            dict_to_add['linker'] = 'drugbank'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['go']['cui']) > 0:
                            dict_to_add = linked_ent['go'].copy()
                            dict_to_add['linker'] = 'go'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['ncbi']['cui']) > 0:
                            dict_to_add = linked_ent['ncbi'].copy()
                            dict_to_add['linker'] = 'ncbi'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['snomed']['cui']) > 0:
                            dict_to_add = linked_ent['snomed'].copy()
                            dict_to_add['linker'] = 'snomed'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['umls']['cui']) > 0:
                            dict_to_add = linked_ent['umls'].copy()
                            dict_to_add['linker'] = 'umls'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        else:
                            sampled_linked_ent.append({})
                    else:
                        if len(linked_ent['hpo']['cui']) > 0:
                            dict_to_add = linked_ent['hpo'].copy()
                            dict_to_add['linker'] = 'hpo'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['go']['cui']) > 0:
                            dict_to_add = linked_ent['go'].copy()
                            dict_to_add['linker'] = 'go'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['rxnorm']['cui']) > 0:
                            dict_to_add = linked_ent['rxnorm'].copy()
                            dict_to_add['linker'] = 'rxnorm'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['gs']['cui']) > 0:
                            dict_to_add = linked_ent['gs'].copy()
                            dict_to_add['linker'] = 'gs'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['drugbank']['cui']) > 0:
                            dict_to_add = linked_ent['drugbank'].copy()
                            dict_to_add['linker'] = 'drugbank'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['mesh']['cui']) > 0:
                            dict_to_add = linked_ent['mesh'].copy()
                            dict_to_add['linker'] = 'mesh'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['ncbi']['cui']) > 0:
                            dict_to_add = linked_ent['ncbi'].copy()
                            dict_to_add['linker'] = 'ncbi'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['snomed']['cui']) > 0:
                            dict_to_add = linked_ent['snomed'].copy()
                            dict_to_add['linker'] = 'snomed'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        elif len(linked_ent['umls']['cui']) > 0:
                            dict_to_add = linked_ent['umls'].copy()
                            dict_to_add['linker'] = 'umls'
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                        else:
                            sampled_linked_ent.append({})

                for ent_sampled in sampled_linked_ent:
                    try:
                        sampled_linked_ent_sub['cui'].extend(ent_sampled['cui'])
                        sampled_linked_ent_sub['name'].extend(ent_sampled['name'])
                        sampled_linked_ent_sub['alias'].extend(ent_sampled['alias'])
                        sampled_linked_ent_sub['tui'].extend(ent_sampled['tui'])
                        sampled_linked_ent_sub['description'].extend(ent_sampled['description'])
                        sampled_linked_ent_sub['probability'].extend(ent_sampled['probability'])
                        sampled_linked_ent_sub['linker'].append(ent_sampled['linker'])
                    except:
                        sampled_linked_ent_sub['cui'].extend([])
                        sampled_linked_ent_sub['name'].extend([])
                        sampled_linked_ent_sub['alias'].extend([])
                        sampled_linked_ent_sub['tui'].extend([])
                        sampled_linked_ent_sub['description'].extend([])
                        sampled_linked_ent_sub['probability'].extend([])
                        sampled_linked_ent_sub['linker'].append('')

                sampled_linked_ent_list.append(sampled_linked_ent_sub)
            data_merged_upd[k1][k2]['sampled_linked_entities'] = sampled_linked_ent_list

    return data_merged_upd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the data retrieval date")
    parser.add_argument("--input_path", default="output/mentions_extraction/", type=str,
                        required=False, help="the path of the files with extracted mentions/entities")

    args = parser.parse_args()

    files = find_json_files(args.input_path + args.date + '/scispacy/merged_entities' + '/')

    for f in files:
        file_name = f.split('/')[-1]

        data = read_json(f)
        data_upd = sampling_linking_codes_strategy(data)

        save_json(data_upd, file_name, args.input_path + args.date + '/scispacy/merged_entities' + '/')