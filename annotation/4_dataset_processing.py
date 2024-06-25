import argparse
import os 
import sys
#from nltk.tokenize import word_tokenize
import scispacy
import spacy
import numpy as np

sys.path.append('../utils/')
from utils import read_json, save_json, find_json_files


nlp = spacy.load("en_core_sci_scibert")
tokenizer = nlp.tokenizer


def tokenize_and_extract_spans(sentence):
    #tokens = word_tokenize(sentence)
    tokens_obj = tokenizer(sentence)
    tokens = []
    for t in tokens_obj:
    	tokens.append(t.text)
    # Corner case
    for i in range(len(tokens)):
    	if tokens[i] == '``' or tokens[i] == "''":
    		tokens[i] = '"'
    spans = []
    for w in tokens:
        if len(spans) == 0:
            spans.append(list(np.arange(0, len(w) + 1)))
        else:
	        if sentence[spans[-1][-1]] == ' ':
	            spans.append(list(np.arange(spans[-1][-1] + 1, spans[-1][-1] + 1 + len(w) + 1)))
	        else:
	            spans.append(list(np.arange(spans[-1][-1], spans[-1][-1] + len(w) + 1)))
    
    return tokens, spans


def get_chunk(pos):
    start = int(pos.split('/')[0]) - 1
    stop = start + int(pos.split('/')[1]) 
    return start, stop


def get_range(pos):
    start = int(pos.split('/')[0]) - 1
    stop = start + int(pos.split('/')[1]) 
    return list(np.arange(start, stop + 1))


def find_start_end_token(position, spans):
    start, end = get_chunk(position)
    flag_start, flag_end = 0, 0
    for i, s in enumerate(spans):
        if s[0] == start:
        	flag_start = 1
        	start_ent = i
        if s[-1] == end:
        	flag_end = 1
        	end_ent = i

    # Flexible matching 
    if flag_start == 0:
    	for i, s in enumerate(spans):
        	if start in s:
        		start_ent = i

    if flag_end == 0:
    	for i, s in enumerate(spans):
        	if end in s:
        		end_ent = i

    return [start_ent, end_ent]    


def add_special_tokens_1(tokens, ranges):
    tokens_updated = []
    ent_1_start, ent_1_end = ranges[0][0], ranges[0][1]
    ent_2_start, ent_2_end = ranges[1][0], ranges[1][1]
    for i, t in enumerate(tokens):
        if i == ent_1_start:
            tokens_updated.append('[ent]')
            tokens_updated.append(t)
            # Corner case: the entity consists of only 1 word
            if ent_1_start == ent_1_end:
                tokens_updated.append('[/ent]')
        elif i == ent_2_start:
            tokens_updated.append('[ent]')
            tokens_updated.append(t)
            # Corner case: the entity consists of only 1 word
            if ent_2_start == ent_2_end:
                tokens_updated.append('[/ent]')
        elif i == ent_1_end:
            tokens_updated.append(t)
            tokens_updated.append('[/ent]')
        elif i == ent_2_end:
            tokens_updated.append(t)
            tokens_updated.append('[/ent]')
        else:
            tokens_updated.append(t)
    return tokens_updated


def add_special_tokens_2(tokens, ranges):
    tokens_updated = []
    ent_1_start, ent_1_end = ranges[0][0], ranges[0][1]
    ent_2_start, ent_2_end = ranges[1][0], ranges[1][1]
    for i, t in enumerate(tokens):
        if i == ent_1_start:
            tokens_updated.append('[ent1]')
            tokens_updated.append(t)
            # Corner case: the entity consists of only 1 word
            if ent_1_start == ent_1_end:
                tokens_updated.append('[/ent1]')
        elif i == ent_2_start:
            tokens_updated.append('[ent2]')
            tokens_updated.append(t)
            # Corner case: the entity consists of only 1 word
            if ent_2_start == ent_2_end:
                tokens_updated.append('[/ent2]')
        elif i == ent_1_end:
            tokens_updated.append(t)
            tokens_updated.append('[/ent1]')
        elif i == ent_2_end:
            tokens_updated.append(t)
            tokens_updated.append('[/ent2]')
        else:
            tokens_updated.append(t)
    return tokens_updated


def update_the_ranges(ranges):
    ent_1_start, ent_1_end = ranges[0][0], ranges[0][1]
    ent_2_start, ent_2_end = ranges[1][0], ranges[1][1]
    if ent_1_start < ent_2_start:
        new_ent_1_start = ent_1_start
        new_ent_1_end = ent_1_end + 2
        new_ent_2_start = ent_2_start + 2
        new_ent_2_end = ent_2_end + 4
    elif ent_1_start > ent_2_start:
        new_ent_2_start = ent_2_start
        new_ent_2_end = ent_2_end + 2
        new_ent_1_start = ent_1_start + 2
        new_ent_1_end = ent_1_end + 4
    else:
        print('Error!')
        
    return [[new_ent_1_start, new_ent_1_end], [new_ent_2_start, new_ent_2_end]]


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--annotator", type=str, required=True, 
                    	help="the name of the annotator")
	parser.add_argument("--disease_name", type=str, required=True, 
						help="the name of the disease")
	parser.add_argument("--dataset_path", default='datasets/', type=str, required=False, 
						help="the path to the merged extracted entities")

	args = parser.parse_args()

	dataset_files = find_json_files(args.dataset_path + args.disease_name + '/' + args.annotator)

	# Merge the datasets
	dataset_total = {}
	for f in dataset_files:
	    dataset = read_json(f)
	    dataset_total.update(dataset)

	for rec in dataset_total:
	    sent = dataset_total[rec]['sentence']
	    tokens, spans = tokenize_and_extract_spans(sent)
	    entities = dataset_total[rec]['entities']
	    range_1 = find_start_end_token(entities[0]['position'], spans)
	    range_2 = find_start_end_token(entities[1]['position'], spans)
	    dataset_total[rec]['entities'][0]['range'] = range_1
	    dataset_total[rec]['entities'][1]['range'] = range_2
	    dataset_total[rec]['tokens'] = tokens
	    dataset_total[rec]['updated_tokens'] = add_special_tokens_1(tokens, [range_1, range_2])
	    updated_ranges = update_the_ranges([range_1, range_2])
	    dataset_total[rec]['entities'][0]['updated_range'] = updated_ranges[0]
	    dataset_total[rec]['entities'][1]['updated_range'] = updated_ranges[1]

	essential_dataset_total = {}
	for rec in dataset_total:
	    essential_dataset_total[rec] = {'sentence': dataset_total[rec]['sentence'],
	                                    'tokens': dataset_total[rec]['tokens'],
	                                    'updated_tokens': dataset_total[rec]['updated_tokens'],
	                                    'entities': [dataset_total[rec]['entities'][0]['range'], dataset_total[rec]['entities'][1]['range']],
	                                    'updated_entities': [dataset_total[rec]['entities'][0]['updated_range'], dataset_total[rec]['entities'][1]['updated_range']],
	                                    'entities_types': [dataset_total[rec]['entities'][0]['mapped_semantic_type'], dataset_total[rec]['entities'][1]['mapped_semantic_type']],
	                                    'relation': dataset_total[rec]['relation']}

	
	save_json(dataset_total, args.dataset_path + args.disease_name + '/' + args.annotator + '/dataset_total.json')		
	save_json(essential_dataset_total, args.dataset_path + args.disease_name + '/' + args.annotator + '/essential_dataset_total.json')		