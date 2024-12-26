import scispacy
import spacy
from scispacy.linking import EntityLinker

class MentionsExtractorSciSpacy:
    def __init__(self, type, linker_type):
        self.type = type
        self.linker_type = linker_type
        self.nlp = self.load_nlp_model()
        self.linker = self.nlp.get_pipe("scispacy_linker")

    def load_nlp_model(self):
        flag = 0
        if self.type == 'craft':
            nlp = spacy.load("en_ner_craft_md")
        elif self.type == 'bc5cdr':
            nlp = spacy.load("en_ner_bc5cdr_md")
        elif self.type == 'jnlpba':
            nlp = spacy.load("en_ner_jnlpba_md")
        elif self.type == 'bionlp13cg':
            nlp = spacy.load("en_ner_bionlp13cg_md")
        else:
            flag == 1
            print('Unknown type given. Supported pipelines: craft, bc5cdr, jnlpba, bionlp13cg')

        if flag == 0:
            # Add the linkers
            if self.linker_type == 'umls':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "umls",
                                                        "threshold": 0.9})
            elif self.linker_type == 'mesh':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "mesh",
                                                        "threshold": 0.9})
            elif self.linker_type == 'rxnorm':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "rxnorm",
                                                        "threshold": 0.9})
            elif self.linker_type == 'go':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "go",
                                                        "threshold": 0.9})
            elif self.linker_type == 'hpo':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "hpo",
                                                        "threshold": 0.9})
            elif self.linker_type == 'drugbank':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "drugbank",
                                                        "threshold": 0.9})
            elif self.linker_type == 'gs':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "gs",
                                                        "threshold": 0.9})
            elif self.linker_type == 'ncbi':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "ncbi",
                                                        "threshold": 0.9})
            elif self.linker_type == 'snomed':
                nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": False,
                                                        "linker_name": "snomed",
                                                        "threshold": 0.9})

            return nlp

    def extract_entities_pos_tags(self, data):
        info = {}
        for id_ in data:
            info[id_] = {}
            for i, s in enumerate(data[id_]['abstract_tokenized']):
                doc = self.nlp(s)

                tokenized_sentence = []
                for token in doc:
                    tokenized_sentence.append(token.text)

                ent_l = []
                linked_l = []
                for ent in doc.ents:
                    ent_l.append((ent.text, ent.label_, ent.start, ent.end, self.type))
                    # Entity linking
                    linked_info = self.get_expanded_entity_linking(ent)
                    linked_l.append(linked_info)

                pos = []
                for token in doc:
                    if token.pos_ in ['NOUN', 'PROPN']:
                        pos.append((token.text, token.pos_))

                #ent_l_unique, linked_l_unique = self.find_unique_entities(ent_l, linked_l)
                info[id_][data[id_]['sentence_ids'][i]] = {'entities': ent_l,
                                                           'linked_entities': linked_l,
                                                           'POS': list(set(pos)),
                                                           'tokenized_sentence': {self.type: tokenized_sentence}}

        return info

    def get_expanded_entity_linking(self, entity):
        linked_info = {}
        cui_l, name_l, aliases_l, tui_l, descr_l, prob_l = self.get_entity_linking(entity)
        linked_info[self.linker_type] = {'cui': cui_l,
                                         'name': name_l,
                                         'alias': aliases_l,
                                         'tui': tui_l,
                                         'description': descr_l,
                                         'probability': prob_l}
        return linked_info

    def get_entity_linking(self, entity):
        cui_l, name_l, aliases_l, tui_l, descr_l, prob_l = [], [], [], [], [], []
        for code_ent in entity._.kb_ents:
            cui_, name_, aliases_, tui_, descr_ = self.linker.kb.cui_to_entity[code_ent[0]]
            cui_l.append(cui_)
            name_l.append(name_)
            aliases_l.append(aliases_)
            tui_l.append(tui_)
            descr_l.append(descr_)
            prob_l.append(code_ent[1])
        return cui_l, name_l, aliases_l, tui_l, descr_l, prob_l

    def find_unique_entities(self, ent_l, linked_l):
        ent_l_unique, linked_l_unique, checked = [], [], []
        for i, en in enumerate(ent_l):
            if (en[0], en[1]) in checked:
                continue
            else:
                ent_l_unique.append(en)
                linked_l_unique.append(linked_l[i])
                checked.append((en[0], en[1]))

        return ent_l_unique, linked_l_unique
