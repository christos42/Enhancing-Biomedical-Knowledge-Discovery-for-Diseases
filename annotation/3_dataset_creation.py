import argparse
import os 
import sys

sys.path.append('../utils/')
from utils import read_json, save_json, find_json_files


def create_dataset(annotations, data_entities, abstracts):
    dataset = {}
    count_rec_per_sent = {}
    for k1 in annotations:
        # Find the keys/indexes for mapping to data_entities
        k2 = '_'.join(k1.split('_')[:2])
        entity_1_index = annotations[k1]['entity_index_pair'][0]
        entity_2_index = annotations[k1]['entity_index_pair'][1]
        all_entities = list(data_entities[k2].keys())
        # Find the keys/indexes for mapping to abstract
        k3 = k2.split('_')[0]
        k4 = int(k2.split('_')[1]) - 1
        sentence = abstracts[k3]['abstract_tokenized'][k4]
        if k2 not in count_rec_per_sent.keys():
            count_rec_per_sent[k2] = 1
        else:
            count_rec_per_sent[k2] += 1
        
        dataset[k2 + '_rec_' + str(count_rec_per_sent[k2])] = {'sentence': sentence,
                                                               'entities': (data_entities[k2][all_entities[entity_1_index]], data_entities[k2][all_entities[entity_2_index]]),
                                                               'relation': annotations[k1]['relation'],
                                                               'useful_text': annotations[k1]['useful text']}

    return dataset


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--entity_path", type=str, required=True, 
						help="the path to the merged extracted entities")
	parser.add_argument("--abstract_path", type=str, required=True, 
						help="the path to the abstract file")
	parser.add_argument("--disease_name", type=str, required=True, 
						help="the name of the disease")
	parser.add_argument("--annotator", type=str, required=True, 
                    	help="the name of the annotator")

	args = parser.parse_args()

	abstracts = read_json(args.abstract_path)
	data_entities = read_json(args.entity_path)
	annotations_files = find_json_files('annotations/' + args.disease_name + '/' + args.annotator + '/')

	output_folder = 'datasets/' + args.disease_name + '/' + args.annotator + '/'
	if not(os.path.isdir(output_folder)):
		os.makedirs(output_folder)

	for f in annotations_files:
		c = f.split('.')[0].split('_')[-1]
		annotations = read_json(f)
		dataset = create_dataset(annotations, data_entities, abstracts)
		save_json(dataset, output_folder + 'dataset_' + c + '.json')		