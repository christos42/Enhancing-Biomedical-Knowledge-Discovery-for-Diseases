from spacy import displacy

def display_ner(nlp, sent):
    doc = nlp(sent)
    displacy.serve(doc, style='ent')

def find_similarity(nlp, en1, en2):
    # word2vec based
    doc1 = nlp(en1)
    doc2 = nlp(en2)
    return doc1.similarity(doc2)

def union_lists(l1, l2):
    l = []
    for it in l1:
        if it not in l:
            l.append(it)
    for it in l2:
        if it not in l:
            l.append(it)

    return l

def union_lists_pairs(l1, l2, l3, l4):
    l_un_1, l_un_2 = [], []
    for i, it in enumerate(l1):
        if it not in l_un_1:
            l_un_1.append(it)
            l_un_2.append(l3[i])
    for i, it in enumerate(l2):
        if it not in l_un_1:
            l_un_1.append(it)
            l_un_2.append(l4[i])

    return l_un_1, l_un_2

def merge_entity_pos_tags_dicts(dict1, dict2):
    dict_ = {}
    for k1 in dict1:
        dict_[k1] = {}
        for k2 in dict1[k1]:
            dict_[k1][k2] = {}
            ent_l_1 = dict1[k1][k2]['entities']
            linked_ent_l_1 = dict1[k1][k2]['linked_entities']
            ent_l_2 = dict2[k1][k2]['entities']
            linked_ent_l_2 = dict2[k1][k2]['linked_entities']
            l_un_1, l_un_2 = union_lists_pairs(ent_l_1, ent_l_2, linked_ent_l_1, linked_ent_l_2)
            dict_[k1][k2]['entities'] = l_un_1
            dict_[k1][k2]['linked_entities'] = l_un_2

            pos_l_1 = dict1[k1][k2]['POS']
            pos_l_2 = dict2[k1][k2]['POS']
            dict_[k1][k2]['POS'] = union_lists(pos_l_1, pos_l_2)

            tokenized_sentence_dict = {}
            for tok1 in dict1[k1][k2]['tokenized_sentence']:
                tokenized_sentence_dict[tok1] = dict1[k1][k2]['tokenized_sentence'][tok1]
            for tok2 in dict2[k1][k2]['tokenized_sentence']:
                tokenized_sentence_dict[tok2] = dict2[k1][k2]['tokenized_sentence'][tok2]

            dict_[k1][k2]['tokenized_sentence'] = tokenized_sentence_dict


    return dict_


def merge_linkers_scispacy(d_umls, d_mesh, d_rxnorm, d_go, d_hpo, d_drugbank, d_gs, d_ncbi, d_snomed):
    d_merged = {}
    for k1 in d_umls:
        d_merged[k1] = {}
        for k2 in d_umls[k1]:
            linked_entities = []
            for en1, en2, en3, en4, en5, en6, en7, en8, en9 in zip(d_umls[k1][k2]['linked_entities'],
                                                                   d_mesh[k1][k2]['linked_entities'],
                                                                   d_rxnorm[k1][k2]['linked_entities'],
                                                                   d_go[k1][k2]['linked_entities'],
                                                                   d_hpo[k1][k2]['linked_entities'],
                                                                   d_drugbank[k1][k2]['linked_entities'],
                                                                   d_gs[k1][k2]['linked_entities'],
                                                                   d_ncbi[k1][k2]['linked_entities'],
                                                                   d_snomed[k1][k2]['linked_entities']):
                linked_entities.append({'umls': en1['umls'],
                                        'mesh': en2['mesh'],
                                        'rxnorm': en3['rxnorm'],
                                        'go': en4['go'],
                                        'hpo': en5['hpo'],
                                        'drugbank': en6['drugbank'],
                                        'gs': en7['gs'],
                                        'ncbi': en8['ncbi'],
                                        'snomed': en9['snomed']})
            d_merged[k1][k2] = {'entities': d_umls[k1][k2]['entities'],
                                'linked_entities': linked_entities,
                                'POS': d_umls[k1][k2]['POS'],
                                'tokenized_sentence': d_umls[k1][k2]['tokenized_sentence']}

    return d_merged


def get_grouped_ne_tag_scispacy(tag):
    tag_grouping = {'CHEMICAL': ['CHEBI', 'CHEMICAL', 'SIMPLE_CHEMICAL'],
                    'CELL': ['CL', 'CELL_TYPE', 'CELL_LINE', 'CELL'],
                    'ORGANISM': ['ORGANISM', 'TAXON'],
                    'GENE_OR_PROTEIN': ['GGP', 'GO', 'PROTEIN', 'GENE_OR_GENE_PRODUCT', 'AMINO_ACID', 'SO']}

    for k in tag_grouping:
        if tag in tag_grouping[k]:
            return k

    return tag


def merge_same_entities_scispacy(data):
    data_entity_merging = {}
    for k1 in data:
        data_entity_merging[k1] = {}
        for k2 in data[k1]:
            ent_dict = {}
            ent_list_triplets, linked_ent_list = [], []
            for i, en in enumerate(data[k1][k2]['entities']):
                if (en[0], en[2], en[3]) not in ent_list_triplets:
                    ent_list_triplets.append((en[0], en[2], en[3]))
                    ent_dict[en[0] + '_' + str(en[2]) + '_' + str(en[3])] = {'type': [en[1]],
                                                                             'grouped_type': [get_grouped_ne_tag_scispacy(en[1])],
                                                                             'pipeline': [en[4]]}
                    linked_ent_list.append(data[k1][k2]['linked_entities'][i])
                else:
                    ent_dict[en[0] + '_' + str(en[2]) + '_' + str(en[3])]['type'].append(en[1])
                    ent_dict[en[0] + '_' + str(en[2]) + '_' + str(en[3])]['grouped_type'].append(get_grouped_ne_tag_scispacy(en[1]))
                    ent_dict[en[0] + '_' + str(en[2]) + '_' + str(en[3])]['pipeline'].append(en[4])
            ent_transformed = []
            for k in ent_dict:
                try:
                    ent_transformed.append({'name': k.split('_')[0],
                                            'start': int(k.split('_')[1]),
                                            'end': int(k.split('_')[2]),
                                            'type': ent_dict[k]['type'],
                                            'grouped_type': ent_dict[k]['grouped_type'],
                                            'pipeline': ent_dict[k]['pipeline']})
                except:
                    ent_transformed.append({'name': k.split('_')[0],
                                            'start': '',
                                            'end': '',
                                            'type': ent_dict[k]['type'],
                                            'grouped_type': ent_dict[k]['grouped_type'],
                                            'pipeline': ent_dict[k]['pipeline']})
            data_entity_merging[k1][k2] = {'entities': ent_transformed,
                                           'linked_entities': linked_ent_list,
                                           'POS': data[k1][k2]['POS'],
                                           'tokenized_sentence': data[k1][k2]['tokenized_sentence']}
    return data_entity_merging