"""Datasets and data loaders of the embedding learning experiments."""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections.abc import Iterable
from typing import Any

from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.training_utils import CV, load_tokenizer_with_markers
from utils.utils import read_json


class Collater:
    """Collate a batch into lists of words, entity ranges, targets and labels."""

    def __init__(self) -> None:
        pass

    def __call__(
        self, data: list[tuple[list[str], list[list[int]], int, int]]
    ) -> list[list[Any]]:
        """Return the words, entity ranges, targets and labels of a batch as lists."""
        words = [item[0] for item in data]
        entities_ranges = [item[1] for item in data]
        targets = [item[2] for item in data]
        relations = [item[3] for item in data]

        return [words, entities_ranges, targets, relations]


class DataProcess(Dataset):
    """Entity pairs with cosine targets (1 if related, else -1) and binary labels.

    Args:
        data: The (tokens with entity markers, entity ranges, relation) examples.
        embed_mode: The backbone (a key of utils.training_utils.BACKBONES), whose
            tokenizer maps the words to sub-words.
    """

    def __init__(
        self, data: list[tuple[list[str], list[list[int]], str]], embed_mode: str
    ) -> None:
        self.data = data
        self.embed_mode = embed_mode
        # Same checkpoint as the model, so that the sub-word offsets match its
        # tokenization
        self.tokenizer = load_tokenizer_with_markers(embed_mode)

        self.mapping_target = {
            "No Relation": -1,
            "Positive Relation": 1,
            "Complex Relation": 1,
            "Negative Relation": 1,
        }

        self.mapping_relation = {
            "No Relation": 0,
            "Positive Relation": 1,
            "Complex Relation": 1,
            "Negative Relation": 1,
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[str], list[list[int]], int, int]:
        words = self.data[idx][0]
        entities_range = self.data[idx][1]
        target = self.mapping_target[self.data[idx][2]]
        relation = self.mapping_relation[self.data[idx][2]]

        # sent_str = ' '.join(words)
        # bert_words = self.tokenizer.tokenize(sent_str)
        # bert_len = original sentence + [CLS] and [SEP]
        # bert_len = len(bert_words) + 2

        word_to_bep = self.map_origin_word_to_bert(words)
        new_entities_range = self.ner_label_transform(entities_range, word_to_bep)

        return (words, new_entities_range, target, relation)

    def map_origin_word_to_bert(self, words: list[str]) -> dict[int, list[int]]:
        """Return the first and last sub-word index of each word.

        The indices exclude the special tokens that the tokenizer adds.
        """
        bep_dict = {}
        current_idx = 0
        for word_idx, word in enumerate(words):
            bert_word = self.tokenizer.tokenize(word)
            word_len = len(bert_word)
            bep_dict[word_idx] = [current_idx, current_idx + word_len - 1]
            current_idx = current_idx + word_len
        return bep_dict

    def ner_label_transform(
        self, entities_range: list[list[int]], word_to_bert: dict[int, list[int]]
    ) -> list[list[int]]:
        """Map word-level entity ranges to positions in the model input.

        The ranges point at the ``[ent]`` and ``[/ent]`` markers, which are single
        sub-words; 1 is added for the leading special token.
        """
        new_entities_range = []
        for r in entities_range:
            # +1 for [CLS]
            new_start = word_to_bert[r[0]][0] + 1
            new_end = word_to_bert[r[1]][0] + 1
            new_entities_range.append([new_start, new_end])

        return new_entities_range


def data_preprocess(
    keys: Iterable[str], data: dict[str, Any]
) -> list[tuple[list[str], list[list[int]], str]]:
    """Return the (tokens with markers, entity ranges, relation) of the records."""
    processed = []
    for k in keys:
        dic = data[k]
        text = dic["updated_tokens"]
        entities = dic["updated_entities"]
        relation = dic["relation"]

        processed += [(text, entities, relation)]
    return processed


def dataloader(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build the training, test and development data loaders.

    The splits come from cross-disease training, cross-validation or the original split.
    """
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
