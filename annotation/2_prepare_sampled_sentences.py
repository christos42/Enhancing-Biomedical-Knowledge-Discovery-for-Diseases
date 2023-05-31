import numpy as np
import argparse
import sys 

sys.path.append('../utils/')
from utils import read_json, save_json

def get_chunk(pos):
    start = int(pos.split('/')[0]) - 1
    stop = start + int(pos.split('/')[1]) 
    return np.arange(start, stop).tolist()


def process_sampled_sentences(sampled_ids, data_entities, abstracts):
	markdown_sentences = {}
	for id_ in sampled_ids:
	    k1 = id_.split('_')[0]
	    index = int(id_.split('_')[1])
	    
	    entities = list(data_entities[id_].keys())
	    sentence = abstracts[k1]['abstract_tokenized'][index-1]
	    
	    for i, ent1 in enumerate(entities):
	        for j, ent2 in enumerate(entities[i+1:]):
	            cui1 = data_entities[id_][ent1]['cui']
	            cui2 = data_entities[id_][ent2]['cui']
	            name1 = data_entities[id_][ent1]['preferred_name']
	            name2 = data_entities[id_][ent2]['preferred_name']
	            if name1 == name2:
	                continue
	            else:
	                cui1_type = " ||| ".join(data_entities[id_][ent1]['mapped_semantic_type'])
	                cui2_type = " ||| ".join(data_entities[id_][ent2]['mapped_semantic_type'])
	                
	                pos1 = get_chunk(data_entities[id_][ent1]['position'])
	                pos2 = get_chunk(data_entities[id_][ent2]['position'])
	                
	                sentence_string_to_present = ''
	                colored_entity_1 = ''
	                colored_entity_2 = ''
	                for c, char in enumerate(sentence):
	                    if c not in pos1 and c not in pos2:
	                        sentence_string_to_present += char
	                    else:
	                    	color_flag_1 = 0
	                    	color_flag_2 = 0
	                    	if c in pos1:
	                    		if char == ' ':
	                    			sentence_string_to_present += char
	                    			colored_entity_1 += char
	                    		else:
	                    			color_flag_1 = 1
	                    			#sentence_string_to_present += ':red[' + char + ']'
	                    			#colored_entity_1 += ':red[' + char + ']'
	                    	if c in pos2:
	                    		if char == ' ':
	                    			sentence_string_to_present += char
	                    			colored_entity_2 += char
	                    		else:
	                    			color_flag_2 = 1
	                    			#sentence_string_to_present += ':blue[' + char + ']'
	                    			#colored_entity_2 += ':blue[' + char + ']'
	                    	if color_flag_1 == 1 and color_flag_2 == 1:
	                    		sentence_string_to_present += ':green[' + char + ']'
	                    		colored_entity_1 += ':green[' + char + ']'
	                    		colored_entity_2 += ':green[' + char + ']'
	                    	elif color_flag_1 == 1:
	                    		sentence_string_to_present += ':red[' + char + ']'
	                    		colored_entity_1 += ':red[' + char + ']'
	                    	elif color_flag_2 == 1:
	                    		sentence_string_to_present += ':blue[' + char + ']'
	                    		colored_entity_2 += ':blue[' + char + ']'     
	                    
	                markdown_sentences[id_ + '_pair_' + str(i) + '_' + str(i+j+1)] = {'sentence': sentence_string_to_present,
                                                                                  	  'cui_type_pair': (cui1_type, cui2_type),
	                                                                          	  	  'entity_pair': (name1, name2),
	                                                                          	  	  'entity_index_pair': (i, j+i+1),
                                                                                  	  'colored_entity_pair': (colored_entity_1, colored_entity_2)}

	return markdown_sentences


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--entity_path", type=str, required=True, 
						help="the path to the merged extracted entities")
	parser.add_argument("--abstract_path", type=str, required=True, 
						help="the path to the abstract file")
	parser.add_argument("--disease_name", type=str, required=True, 
						help="the name of the disease")
	parser.add_argument("--strategy_id", type=str, required=True, 
						help="the sampling strategy id, supported values: 1, 2")
	parser.add_argument("--bucket_id", type=int, required=True, 
						help="the id number of the bucket")

	args = parser.parse_args()

	abstracts = read_json(args.abstract_path)
	data_entities = read_json(args.entity_path)
	sampled_ids = read_json('strategy_' + str(args.strategy_id) + '/' + args.disease_name + '/bucket_' + str(args.bucket_id) + '.json')

	markdown_sentences = process_sampled_sentences(sampled_ids, data_entities, abstracts)
	
	save_json(markdown_sentences, 'markdown_sentences/' + args.disease_name + '/markdown_sentences_' + str(args.bucket_id) + '.json')
	