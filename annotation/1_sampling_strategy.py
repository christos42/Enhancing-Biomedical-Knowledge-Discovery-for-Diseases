import argparse
import numpy as np 
import os 
import sys 
import random

sys.path.append('../utils/')
from utils import read_json, save_json

# function that creates random sample 
def random_sampling(ids, n):
    random_sample = np.random.choice(ids, replace = False, size = n)
    return(random_sample)

def weighted_random_sampling(ids, prob, n):
    random_sample = rng.choice(ids, replace = False, size = n, p = prob, shuffle=False)
    return list((random_sample))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_path", type=str, required=True, 
                        help="the path to the merged extracted entities")
    parser.add_argument("--cooccurrence_path", type=str, required=True, 
                        help="the path to the co-occurrence graph")
    parser.add_argument("--abstract_path", type=str, required=True, 
                        help="the path to the abstract file")
    parser.add_argument("--disease_name", type=str, required=True, 
                        help="the name of the disease")
    parser.add_argument("--strategy_id", type=str, required=True, 
                        help="the sampling strategy id, supported values: 1, 2")
    parser.add_argument("--n", type=int, default=25, required=False,
                        help="the number of sampled sentences per bucket (sampling strategy: 1)")
    parser.add_argument("--n_conc", type=int, default=10, required=False,
                        help="the number of concept pairs to be sampled per bucket (sampling strategy: 2)")

    args = parser.parse_args()

    rng = np.random.default_rng()
    
    # Read the files
    data_entities = read_json(args.entity_path)
    cooc = read_json(args.cooccurrence_path)
    abstracts = read_json(args.abstract_path)

    n = args.n 

    print('Number of concept pairs:', format(len(cooc.keys())))
    
    
    # Find ids with atleast 2 mapped entities
    sampled_ids = []
    for k1 in data_entities:
        if len(list(data_entities[k1].keys())) >= 2:
            sampled_ids.append(k1)
    print('Find ids with atleast 2 mapped entities: DONE')

    freq_pairs_per_sent = {}
    for s in sampled_ids:
        ent_info = data_entities[s]
        cuis = []
        for ent in ent_info:
            cuis.append(ent_info[ent]['cui'])
        
        cui_pairs = []
        for i1, cui1 in enumerate(cuis):
            for cui2 in cuis[i1+1:]:
                if cui1 != cui2:
                    cui_pairs.append((cui1, cui2))
        
        freq_per_pair = []
        for p in cui_pairs:
            cui_co_1 = p[0].replace('||', '_')
            cui_co_2 = p[1].replace('||', '_')
            try:
                freq_per_pair.append(cooc[cui_co_1 + '_' + cui_co_2]['frequency'])
            except:
                freq_per_pair.append(cooc[cui_co_2 + '_' + cui_co_1]['frequency'])
        
        if len(freq_per_pair) > 0:
            freq_pairs_per_sent[s] = {'concept pairs': cui_pairs,
                                      'frequency per pair': freq_per_pair,
                                      'total frequency': int(np.sum(freq_per_pair))}

    freq_pairs_per_sent_sorted = dict(sorted(freq_pairs_per_sent.items(), 
                                             key=lambda item: item[1]['total frequency'], 
                                             reverse=True))
    print('Create dictionary with frequency information: DONE')
    
    #save_json(freq_pairs_per_sent_sorted, 'freq_pairs_per_sent_sorted.json')
    #freq_pairs_per_sent_sorted = read_json('freq_pairs_per_sent_sorted.json')

    ids, total_freq, inversed_total_freq = [], [], []
    for k1 in freq_pairs_per_sent_sorted:
        ids.append(k1)
        total_freq.append(freq_pairs_per_sent_sorted[k1]['total frequency'])
        inversed_total_freq.append(1/freq_pairs_per_sent_sorted[k1]['total frequency'])
    print('Frequencies: DONE')

    counter = 0
    probabilities = []
    total_freq_sum = np.sum(total_freq)
    for f in total_freq:
        probabilities.append(float(f/total_freq_sum))
        counter += 1
        if counter % 50000 == 0:
            print('{} probabilities counted.' .format(counter))
            save_json(probabilities, 'probabilities.json')
    print('Create probabilities based on frequencies: DONE')
    #save_json(probabilities, 'probabilities.json')
    #probabilities = read_json('probabilities.json')
    
    counter = 0
    inversed_probabilities = []
    inversed_total_freq_sum = np.sum(inversed_total_freq)
    for f in inversed_total_freq:
        inversed_probabilities.append(float(f/inversed_total_freq_sum))
        counter += 1
        if counter % 50000 == 0:
            print('{} inversed probabilities counted.' .format(counter))
            save_json(inversed_probabilities, 'inversed_probabilities.json')
    print('Create inversed probabilities based on frequencies: DONE')
    #save_json(inversed_probabilities, 'inversed_probabilities.json')
    #inversed_probabilities = read_json('inversed_probabilities.json')


    counter = 0
    if args.strategy_id == '1':
        folder = 'strategy_1/' + args.disease_name + '/'
        if not(os.path.isdir(folder)):
            os.makedirs(folder)

        buckets = []
        ids_to_be_sampled = ids.copy()
        probabilities_to_be_sampled = probabilities.copy()
        inversed_probabilities_to_be_sampled = inversed_probabilities.copy()

        # Add the accumulated artificial id: Probabilities should sum to 1.
        ids_to_be_sampled.append('foo')
        probabilities_to_be_sampled.append(0)
        inversed_probabilities_to_be_sampled.append(0)

        while(len(ids_to_be_sampled) >= 50):
            b1 = weighted_random_sampling(ids_to_be_sampled, probabilities_to_be_sampled, n)
            try:
                # Remove artificial id in case it was sampled.
                b1.remove('foo')
            except:
                pass
            # Remove the sampled ids from the list
            for id_ in b1:
                index = ids_to_be_sampled.index(id_)
                del ids_to_be_sampled[index]
                probabilities_to_be_sampled[-1] += probabilities_to_be_sampled[index]
                del probabilities_to_be_sampled[index]
                inversed_probabilities_to_be_sampled[-1] +=  inversed_probabilities_to_be_sampled[index]
                del inversed_probabilities_to_be_sampled[index]
            
            b2 = weighted_random_sampling(ids_to_be_sampled, inversed_probabilities_to_be_sampled, n)
            try:
                # Remove artificial id in case it was sampled.
                b2.remove('foo')
            except:
                pass
            # Remove the sampled ids from the list
            for id_ in b2:
                index = ids_to_be_sampled.index(id_)
                del ids_to_be_sampled[index]
                probabilities_to_be_sampled[-1] += probabilities_to_be_sampled[index]
                del probabilities_to_be_sampled[index]
                inversed_probabilities_to_be_sampled[-1] +=  inversed_probabilities_to_be_sampled[index]
                del inversed_probabilities_to_be_sampled[index]
            
            #buckets.append(random.shuffle(list(b1) + list(b2)))
            save_json(random.shuffle(list(b1) + list(b2)), 'bucket_' + str(counter) + '.json', folder)
            counter += 1

        try:
            # Remove artificial id.
            ids_to_be_sampled.remove('foo')
        except:
            pass
            
        if len(ids_to_be_sampled) > 0:
            buckets.append(ids_to_be_sampled)


        #folder = 'strategy_1/' + args.disease_name + '/'
        #if not(os.path.isdir(folder)):
        #    os.makedirs(folder)

        #for i, b in enumerate(buckets):
        #    save_json(b, 'bucket_' + str(i) + '.json', folder)
    elif args.strategy_id == '2':
        concept_pairs = list(cooc.keys())
        buckets_2 = []
        concept_pairs_to_be_sampled = concept_pairs.copy()
        all_sampled_ids = []

        sampled_ids = []
        while(len(concept_pairs_to_be_sampled) >= 10):    
            b1 = random_sampling(concept_pairs_to_be_sampled, args.n_conc)
            for k in b1:
                for id_ in cooc[k]['sentence_ids']:
                    if id_ not in all_sampled_ids:
                        all_sampled_ids.append(id_)
                        sampled_ids.append(id_)
                index = concept_pairs_to_be_sampled.index(k)
                del concept_pairs_to_be_sampled[index]
            
            step = 50
            if len(sampled_ids) >= step:
                buckets_2.append(random.shuffle(sampled_ids[:step]))
                sampled_ids = sampled_ids[step:]

        for k in concept_pairs_to_be_sampled:
            for id_ in cooc[k]['sentence_ids']:
                if id_ not in all_sampled_ids:
                    all_sampled_ids.append(id_)
                    sampled_ids.append(id_)

        for i in range(0, len(sampled_ids), step):
            try:
                buckets_2.append(random.shuffle(sampled_ids[:step]))
            except:
                buckets_2.append(random.shuffle(sampled_ids[step:]))
        folder = 'strategy_2/' + args.disease_name + '/'
        if not(os.path.isdir(folder)):
            os.makedirs(folder)

        for i, b in enumerate(buckets_2):
            save_json(b, 'bucket_' + str(i) + '.json', folder)
    else:
        print('Unsupported sampling strategy id is given. Supported values: 1 and 2.')

    
    # Create the necessary folders
    folder = 'annotations/' + args.disease_name + '/' 
    if not(os.path.isdir(folder)):
            os.makedirs(folder)

    folder = 'markdown_sentences/' + args.disease_name + '/' 
    if not(os.path.isdir(folder)):
            os.makedirs(folder)

    folder = 'entities_to_be_removed/' + args.disease_name + '/' 
    if not(os.path.isdir(folder)):
            os.makedirs(folder)

    folder = 'sentences_to_be_removed/' + args.disease_name + '/' 
    if not(os.path.isdir(folder)):
            os.makedirs(folder)
