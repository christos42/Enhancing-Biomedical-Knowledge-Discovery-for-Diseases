import os
import random
import sys

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from utils.training_utils import CV, PROBING_BACKBONES
from utils.utils import read_json


class Collater:
    def __init__(self):
        pass

    def __call__(self, data):
        words = [item[0] for item in data]
        entities_ranges = [item[1] for item in data]
        relations = [item[2] for item in data]

        return [words, entities_ranges, relations]


class DataProcess(Dataset):
    def __init__(self, data, embed_mode, exp_setting):
        self.data = data
        self.embed_mode = embed_mode
        self.tokenizer = AutoTokenizer.from_pretrained(PROBING_BACKBONES[embed_mode][0])

        if exp_setting == "binary":
            self.mapping = {
                "No Relation": 0,
                "Positive Relation": 1,
                "Complex Relation": 1,
                "Negative Relation": 1,
            }
        elif exp_setting == "multi_class":
            self.mapping = {
                "No Relation": 0,
                "Positive Relation": 1,
                "Complex Relation": 2,
                "Negative Relation": 3,
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data[idx][0]
        entities_range = self.data[idx][1]
        relation = self.mapping[self.data[idx][2]]

        # sent_str = ' '.join(words)
        # bert_words = self.tokenizer.tokenize(sent_str)
        # bert_len = original sentence + [CLS] and [SEP]
        # bert_len = len(bert_words) + 2

        word_to_bep = self.map_origin_word_to_bert(words)
        new_entities_range = self.ner_label_transform(entities_range, word_to_bep)

        return (words, new_entities_range, relation)

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
            # Last sub-word of the entity's last word (there are no entity markers here)
            new_end = word_to_bert[r[1]][1] + 1
            new_entities_range.append([new_start, new_end])

        return new_entities_range


def data_preprocess(keys, data):
    processed = []
    for k in keys:
        dic = data[k]
        text = dic["tokens"]
        entities = dic["entities"]
        relation = dic["relation"]

        processed += [(text, entities, relation)]
    return processed


def dataloader(args):
    data = read_json(args.dataset_path)
    # Create the fold of keys for training and test (5-fold CV is applied)
    keys = list(data.keys())
    if args.do_cross_disease_training:
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

    train_dataset = DataProcess(train_data, args.embed_mode, args.exp_setting)
    test_dataset = DataProcess(test_data, args.embed_mode, args.exp_setting)
    dev_dataset = DataProcess(dev_data, args.embed_mode, args.exp_setting)

    collate_fn = Collater()

    train_batch = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_batch = DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dev_batch = DataLoader(
        dataset=dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_batch, test_batch, dev_batch
