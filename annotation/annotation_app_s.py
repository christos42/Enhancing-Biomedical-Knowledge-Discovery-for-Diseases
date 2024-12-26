import streamlit as st
import argparse
import random
import sys 
import os
from datetime import datetime

sys.path.append('../utils/')
from utils import read_json, save_json


state = st.session_state

parser = argparse.ArgumentParser()
parser.add_argument("--bucket_id", type=int, required=True, 
                    help="the id number of the bucket")
parser.add_argument("--trial_id", type=int, required=True,
                    help="the id number of the trial")
parser.add_argument("--disease_name", type=str, required=True, 
                    help="the name of the disease")
parser.add_argument("--annotator", type=str, required=True, 
                    help="the name of the annotator")

args = parser.parse_args()


# Getting the current date and time
dt = datetime.now()

# getting the timestamp
ts = str(datetime.timestamp(dt))
ts = ts.replace('.', '_')

markdown_sentences = read_json('markdown_sentences/' + args.disease_name + '/markdown_sentences_' + str(args.bucket_id) + '.json') 

# Create the output path if needed
folder_1 = 'annotations/' + args.disease_name + '/' + args.annotator + '/'
if not(os.path.isdir(folder_1)):
  os.makedirs(folder_1)
folder_2 = 'entities_to_be_removed/' + args.disease_name + '/' + args.annotator + '/'
if not(os.path.isdir(folder_2)):
  os.makedirs(folder_2)
folder_3 = 'sentences_to_be_removed/' + args.disease_name + '/' + args.annotator + '/'
if not(os.path.isdir(folder_3)):
  os.makedirs(folder_3)


OPTIONS_RELATION = ["Positive Relation", "Negative Relation", "Complex Relation", "No Relation"]
LEN_OPTIONS_RELATION = [len(s) for s in OPTIONS_RELATION]

if "annotations" not in state:
  state.annotations = {}
  state.sentence_ids = list(markdown_sentences.keys())
  state.current_sentence_id = state.sentence_ids[0]
  state.annotations[state.current_sentence_id] = {'entity_pair': markdown_sentences[state.current_sentence_id]['entity_pair'],
                                                  'entity_index_pair': markdown_sentences[state.current_sentence_id]['entity_index_pair']}
  #state.sentence_ids.remove(state.current_sentence_id)

if "entities_to_be_removed" not in state:
  state.entities_to_be_removed = {}

if "sentences_to_be_removed" not in state:
  state.sentences_to_be_removed = []


def annotation_accumulation(flag, label):
  if flag == "1":
    state.annotations[state.current_sentence_id]['relation'] = label
  elif flag == "2":
    state.annotations[state.current_sentence_id]['useful text'] = label

  added_information = list(state.annotations[state.current_sentence_id].keys())
  if 'relation' in added_information and 'useful text' in added_information:
    state.sentence_ids.remove(state.current_sentence_id)
    st.session_state["foo"] = ""
    if state.sentence_ids:
      state.current_sentence_id = state.sentence_ids[0]
      state.annotations[state.current_sentence_id] = {'entity_pair': markdown_sentences[state.current_sentence_id]['entity_pair'],
                                                      'entity_index_pair': markdown_sentences[state.current_sentence_id]['entity_index_pair']}


def remove_entity(entity_index):
  sentence_id = '_'.join(state.current_sentence_id.split('_')[:2])
  if sentence_id not in state.entities_to_be_removed.keys():
    state.entities_to_be_removed[sentence_id] = [entity_index]
  else:
    state.entities_to_be_removed[sentence_id].append(entity_index)
  # Remove the concept pairs that contain this entity
  pairs_to_remove = []
  for id_ in state.sentence_ids:
    tmp_sentence_id = '_'.join(id_.split('_')[:2])
    tmp_entity_index_1 = id_.split('_')[3]
    tmp_entity_index_2 = id_.split('_')[4]
    if sentence_id == tmp_sentence_id:
      if entity_index in [tmp_entity_index_1, tmp_entity_index_2]:
        pairs_to_remove.append(id_)
  for pair_to_remove in pairs_to_remove:
    state.sentence_ids.remove(pair_to_remove)
  # Continue with the next pair
  if state.sentence_ids:
    # Remove the current concept pair from the annotations.
    state.annotations.pop(state.current_sentence_id)
    state.current_sentence_id = state.sentence_ids[0]
    state.annotations[state.current_sentence_id] = {'entity_pair': markdown_sentences[state.current_sentence_id]['entity_pair'],
                                                    'entity_index_pair': markdown_sentences[state.current_sentence_id]['entity_index_pair']}


def remove_sentence(sentence_id_r):
  state.sentences_to_be_removed.append(sentence_id_r)
  pairs_to_remove = []
  for id_ in state.sentence_ids:
    tmp_sentence_id = '_'.join(id_.split('_')[:2])
    if sentence_id_r == tmp_sentence_id:
      pairs_to_remove.append(id_)
  for pair_to_remove in pairs_to_remove:
    state.sentence_ids.remove(pair_to_remove)
  # Continue with the next pair
  if state.sentence_ids:
    # Remove the current concept pair from the annotations.
    state.annotations.pop(state.current_sentence_id)
    state.current_sentence_id = state.sentence_ids[0]
    state.annotations[state.current_sentence_id] = {'entity_pair': markdown_sentences[state.current_sentence_id]['entity_pair'],
                                                    'entity_index_pair': markdown_sentences[state.current_sentence_id]['entity_index_pair']}



st.header("Dataset annotation")


if state.sentence_ids:
  #st.text('lala')
  sentence_id = state.current_sentence_id
  entity_1_name, entity_2_name = markdown_sentences[sentence_id]['colored_entity_pair']
  entity_1_umls_type, entity_2_umls_type = markdown_sentences[sentence_id]['cui_type_pair']

  st.markdown(markdown_sentences[sentence_id]['sentence'])

  st.write('----------------------------------------------')
  sentence_id_displ = 'Sentence id: ' + '_'.join(sentence_id.split('_')[:2])
  st.write(sentence_id_displ)
  st.button('Remove the sentence.', type ='secondary', on_click=remove_sentence, args=('_'.join(sentence_id.split('_')[:2]), ))
  st.write('----------------------------------------------')
  entity_index_1 = sentence_id.split('_')[3]
  entity_index_2 = sentence_id.split('_')[4]
  displ1 = entity_1_name + ': ' + entity_1_umls_type
  st.button('Remove the first entity.', type ='secondary', on_click=remove_entity, args=(entity_index_1, ))
  st.write(displ1)
  displ2 = entity_2_name + ': ' + entity_2_umls_type
  st.write(displ2)
  st.button('Remove the second entity.', type ='secondary', on_click=remove_entity, args=(entity_index_2, ))
  st.write('----------------------------------------------')

  st.write('Define the relation between the colored concepts/entities: {} & {}' .format(entity_1_name, entity_2_name))
  c1 = st.columns(len(OPTIONS_RELATION))
  for idx, option1 in enumerate(OPTIONS_RELATION):
    c1[idx].button(f"{option1}", on_click=annotation_accumulation, args=("1", option1, ))

  st.write('----------------------------------------------')
  text_input = st.text_input(
        label = "Useful piece of text that helped to define the relation: ",
        key="foo",
        label_visibility="visible"
    )
  st.write('----------------------------------------------')  
  st.button('Done!', type ='primary', on_click=annotation_accumulation, args=("2", text_input, ))

  # Constantly save the annotations
  save_json(state.annotations, folder_1 + 'annotations_' + str(args.bucket_id) + '_' + str(args.trial_id) + '_' + ts + '.json')
  save_json(state.entities_to_be_removed, folder_2 + 'entities_to_be_removed_' + str(args.bucket_id) + '_' + str(args.trial_id) + '_' + ts +  '.json')
  save_json(state.sentences_to_be_removed, folder_3 + 'sentences_to_be_removed_' + str(args.bucket_id) + '_' + str(args.trial_id) + '_' + ts + '.json')

else:
  st.info("Everything annotated.")

  # Final cross-check
  pairs_to_remove = []
  for p_id in state.annotations:
    s_id = '_'.join(p_id.split('_')[:2])
    if s_id in state.sentences_to_be_removed:
      pairs_to_remove.append(p_id)
    if s_id in state.entities_to_be_removed:
      entity_index_1 = str(state.annotations[p_id]['entity_index_pair'][0])
      entity_index_2 = str(state.annotations[p_id]['entity_index_pair'][1])
      if entity_index_1 in state.entities_to_be_removed[s_id] or entity_index_2 in state.entities_to_be_removed[s_id]:
        pairs_to_remove.append(p_id)

  pairs_to_remove = list(set(pairs_to_remove))

  for p_to_remove in pairs_to_remove:
    state.annotations.pop(p_to_remove)

  save_json(state.annotations, folder_1 + 'annotations_' + str(args.bucket_id) + '_' + str(args.trial_id) + '_' + ts + '.json')


st.info(f"Annotated: {len(state.annotations)}, Remaining: {len(state.sentence_ids)}")

