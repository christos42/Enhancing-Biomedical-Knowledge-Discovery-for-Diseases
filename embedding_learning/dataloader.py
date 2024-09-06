import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.utils.data import Dataset, DataLoader
import random

sys.path.append('../')
from utils.utils import read_json

class collater_1():
    def __init__(self):
        pass

    def __call__(self, data):
        words = [item[0] for item in data]
        entities_ranges = [item[1] for item in data]
        targets = [item[2] for item in data]
        relations = [item[3] for item in data]

        return [words, entities_ranges, targets, relations]


class DataProcess(Dataset):
    def __init__(self, data, embed_mode):
        self.data = data
        self.embed_mode = embed_mode
        if embed_mode == 'BiomedBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            # Add the special tokens [ent] & [/ent] in the vocabulary
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
        elif embed_mode == 'BiomedBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract")
            # Add the special tokens [ent] & [/ent] in the vocabulary
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
        elif embed_mode == 'BioLinkBERT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
            # Add the special tokens [ent] & [/ent] in the vocabulary
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
        elif embed_mode == 'BioLinkBERT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-large")
            # Add the special tokens [ent] & [/ent] in the vocabulary
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
        elif embed_mode == 'BioGPT_base':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
            # Add the special tokens [ent] & [/ent] in the vocabulary
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])
        elif embed_mode == 'BioGPT_large':
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
            # Add the special tokens [ent] & [/ent] in the vocabulary
            self.tokenizer.add_tokens(['[ent]'])
            self.tokenizer.add_tokens(['[/ent]'])


        self.mapping_target = {'No Relation': -1,
                               'Positive Relation': 1,
                               'Complex Relation': 1,
                               'Negative Relation': 1}

        self.mapping_relation = {'No Relation': 0,
                                 'Positive Relation': 1,
                                 'Complex Relation': 1,
                                 'Negative Relation': 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx][0]
        entities_range = self.data[idx][1]
        target = self.mapping_target[self.data[idx][2]]
        relation = self.mapping_relation[self.data[idx][2]]

        #sent_str = ' '.join(words)
        #bert_words = self.tokenizer.tokenize(sent_str)
        # bert_len = original sentence + [CLS] and [SEP]
        #bert_len = len(bert_words) + 2

        word_to_bep = self.map_origin_word_to_bert(words)
        new_entities_range = self.ner_label_transform(entities_range, word_to_bep)

        return (words, new_entities_range, target, relation)


    def map_origin_word_to_bert(self, words):
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform(self, entities_range, word_to_bert):
        new_entities_range = []
        for r in entities_range:
            # +1 for [CLS]
            new_start = word_to_bert[r[0]][0] + 1
            new_end = word_to_bert[r[1]][0] + 1
            new_entities_range.append([new_start, new_end])

        return new_entities_range


def data_preprocess(keys, data):
    processed = []
    for k in keys:
        dic = data[k]
        text = dic['updated_tokens']
        entities = dic['updated_entities']
        relation = dic['relation']

        processed += [(text, entities, relation)]
    return processed


class CV():
    def __init__(self, keys, k):
        self.keys = keys
        self.k = k

    def get_cv_splits(self, fold):
        splits = []
        step = len(self.keys) // self.k
        for i in range(0, self.k * step, step):
            splits.append(self.keys[i:i + step])
        # Add the remaining keys in the last fold
        splits[-1] += self.keys[self.k * step:]
        # k-fold CV
        train_keys = []
        test_keys = []
        for i, s in enumerate(splits):
            if i == fold:
                test_keys += s
            else:
                train_keys += s
        return train_keys, test_keys

    def get_unique_sentences(self):
        sentences = []
        for k in self.keys:
            if '_'.join(k.split('_')[:2]) not in sentences:
                sentences.append('_'.join(k.split('_')[:2]))
        return sentences

    def get_cv_splits_sentence_wise(self, fold):
        sentences = self.get_unique_sentences()
        splits = []
        step = len(sentences) // self.k
        for i in range(0, self.k * step, step):
            splits.append(sentences[i:i + step])
        # Add the remaining sentences in the last fold
        splits[-1] += sentences[self.k * step:]
        splits_keys = []
        for i, s in enumerate(splits):
            temp_s = []
            for sent in s:
                for k in self.keys:
                    if '_'.join(k.split('_')[:2]) == sent:
                        temp_s.append(k)
            splits_keys.append(temp_s)
        # 5 fold CV
        train_keys = []
        test_keys = []
        for i, s in enumerate(splits_keys):
            if i == fold:
                test_keys += s
            else:
                train_keys += s
        return train_keys, test_keys


def dataloader(args):
    if args.do_cross_disease_training:
        data = read_json(args.dataset_path)
        # Create the fold of keys for training and test (5-fold CV is applied)
        keys = list(data.keys())
        random.seed(args.seed)
        random.shuffle(keys)
        split = int(0.15 * len(keys))
        train_keys = keys[split:]
        dev_keys = keys[:split]
        # Test/Evaluation data
        data_test = read_json(args.dataset_path_eval)
        test_keys = list(data_test.keys())
        # Preprocess the data before sending them in the Dataset class
        train_data = data_preprocess(train_keys, data)
        test_data = data_preprocess(test_keys, data_test)
        dev_data = data_preprocess(dev_keys, data)
    else:
        if args.do_cross_validation:
            data = read_json(args.dataset_path)
            # Create the fold of keys for training and test (5-fold CV is applied)
            keys = list(data.keys())
            cv = CV(keys, 5)
            if args.sentence_wise_splits:
                train_keys_all, test_keys = cv.get_cv_splits_sentence_wise(args.fold)
            else:
                train_keys_all, test_keys = cv.get_cv_splits(args.fold)
            random.seed(42)
            random.shuffle(train_keys_all)
            split = int(0.15 * len(train_keys_all))
            train_keys = train_keys_all[split:]
            dev_keys = train_keys_all[:split]
            # Preprocess the data before sending them in the Dataset class
            train_data = data_preprocess(train_keys, data)
            test_data = data_preprocess(test_keys, data)
            dev_data = data_preprocess(dev_keys, data)
        else:
            dataset_train = read_json(args.dataset_path_train)
            dataset_dev = read_json(args.dataset_path_dev)
            dataset_test = read_json(args.dataset_path_test)

            train_keys = list(dataset_train.keys())
            random.seed(args.seed)
            random.shuffle(train_keys)

            # Preprocess the data before sending them in the Dataset class
            train_data = data_preprocess(train_keys, dataset_train)
            test_data = data_preprocess(dataset_test.keys(), dataset_test)
            dev_data = data_preprocess(dataset_dev.keys(), dataset_dev)

    train_dataset = DataProcess(train_data, args.embed_mode)
    test_dataset = DataProcess(test_data, args.embed_mode)
    dev_dataset = DataProcess(dev_data, args.embed_mode)

    collate_fn = collater_1()

    train_batch = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                             collate_fn=collate_fn)
    test_batch = DataLoader(dataset=test_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True,
                            collate_fn=collate_fn)
    dev_batch = DataLoader(dataset=dev_dataset, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True,
                           collate_fn=collate_fn)

    return train_batch, test_batch, dev_batch