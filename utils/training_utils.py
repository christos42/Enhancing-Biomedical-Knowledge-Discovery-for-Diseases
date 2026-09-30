"""Code shared by the relation detection, probing and embedding learning experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# Hugging Face checkpoint and hidden size of each backbone (--embed_mode)
BACKBONES = {
    "BiomedBERT_base": ("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", 768),
    "BiomedBERT_large": ("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract", 1024),
    "BioLinkBERT_base": ("michiyasunaga/BioLinkBERT-base", 768),
    "BioLinkBERT_large": ("michiyasunaga/BioLinkBERT-large", 1024),
    "BioGPT_base": ("microsoft/biogpt", 1024),
    "BioGPT_large": ("microsoft/BioGPT-Large", 1600),
}

# Backbones of the probing experiments, which use no entity markers
PROBING_BACKBONES = {
    "PubMedBERT_base": (
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        768,
    ),
    "PubMedBERT_large": ("microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract", 1024),
}


def load_tokenizer_with_markers(embed_mode: str) -> PreTrainedTokenizerBase:
    """Load a backbone's tokenizer and add the entity markers ``[ent]`` and ``[/ent]``.

    Args:
        embed_mode: The backbone (a key of BACKBONES).
    """
    tokenizer = AutoTokenizer.from_pretrained(BACKBONES[embed_mode][0])
    tokenizer.add_tokens(["[ent]"])
    tokenizer.add_tokens(["[/ent]"])
    return tokenizer


def load_backbone_with_markers(
    embed_mode: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, int]:
    """Load a backbone's tokenizer and language model, with the entity markers added.

    The embeddings of ``[ent]`` and ``[/ent]`` are initialized with small random values,
    drawn with a fixed seed that is independent of ``--seed``.

    Args:
        embed_mode: The backbone (a key of BACKBONES).

    Returns:
        The tokenizer, the language model and its hidden size.
    """
    checkpoint, hidden_size = BACKBONES[embed_mode]
    tokenizer = load_tokenizer_with_markers(embed_mode)
    model = AutoModel.from_pretrained(checkpoint)
    is_gpt = embed_mode.startswith("BioGPT")
    weights = (
        model.embed_tokens if is_gpt else model.embeddings.word_embeddings
    ).weight.data
    # Own generator, so --seed still drives everything else
    generator = torch.Generator().manual_seed(42)
    # Idea: small initialization embedding
    w1 = torch.unsqueeze(
        torch.empty(hidden_size).uniform_(-1e-4, 1e-4, generator=generator), 0
    )
    w2 = torch.unsqueeze(
        torch.empty(hidden_size).uniform_(-1e-4, 1e-4, generator=generator), 0
    )
    new_weights = torch.cat((weights, w1, w2), 0)
    # Also place them at the ids the tokenizer assigned, which precede the appended rows
    # when its vocabulary is smaller than the matrix
    new_weights[tokenizer.convert_tokens_to_ids(["[ent]", "[/ent]"])] = torch.cat(
        (w1, w2), 0
    )
    new_emb = torch.nn.Embedding.from_pretrained(
        new_weights, padding_idx=0, freeze=False
    )
    if is_gpt:
        model.embed_tokens = new_emb
    else:
        model.embeddings.word_embeddings = new_emb
    return tokenizer, model, hidden_size


def load_frozen_backbone(
    embed_mode: str,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, int]:
    """Load a probing backbone's tokenizer and language model, with its layers frozen.

    Args:
        embed_mode: The backbone (a key of PROBING_BACKBONES).

    Returns:
        The tokenizer, the language model and its hidden size.
    """
    checkpoint, hidden_size = PROBING_BACKBONES[embed_mode]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    # Freeze the encoding layers
    for module in [model.embeddings, *model.encoder.layer]:
        for param in module.parameters():
            param.requires_grad = False
    return tokenizer, model, hidden_size


def inter_representation(
    r1: torch.Tensor, start_ent_1: int, end_ent_1: int, start_ent_2: int, end_ent_2: int
) -> torch.Tensor:
    """Return the mean representation of the tokens between two entities.

    When nothing is between them (adjacent or overlapping entities), the mean of their
    start tokens is used.

    Args:
        r1: The token representations of the sentence (tokens x hidden size).
        start_ent_1: The first token of the first entity.
        end_ent_1: The last token of the first entity.
        start_ent_2: The first token of the second entity.
        end_ent_2: The last token of the second entity.
    """
    if end_ent_1 + 1 == start_ent_2:
        return torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
    elif end_ent_2 + 1 == start_ent_1:
        return torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)
    elif end_ent_1 < start_ent_2:
        return torch.mean(r1[end_ent_1 + 1 : start_ent_2], 0)
    elif end_ent_2 < start_ent_1:
        return torch.mean(r1[end_ent_2 + 1 : start_ent_1], 0)
    return torch.mean(torch.stack([r1[start_ent_1], r1[start_ent_2]]), 0)


def atlop_context_vector(
    attentions: torch.Tensor,
    r1: torch.Tensor,
    head_span: tuple[int, int],
    tail_span: tuple[int, int],
) -> torch.Tensor:
    """Return the ATLOP-style context vector of an entity pair.

    The token representations are averaged, weighted by the product of the two entities'
    attention to each token (averaged over the heads and normalized).

    Args:
        attentions: One example's attention scores of a layer (heads x tokens x tokens).
        r1: The example's token representations (tokens x hidden size).
        head_span: The inclusive (start, end) positions of the attention rows of the
            first entity.
        tail_span: The same, for the second entity.
    """
    # extract attentions of the two entities and sequence
    head_attentions = torch.mean(attentions[:, head_span[0] : head_span[1] + 1, :], 1)
    tail_attentions = torch.mean(attentions[:, tail_span[0] : tail_span[1] + 1, :], 1)

    # hadamard product of the head_attentions and tail_attentions, then average over
    # heads
    head_tail_attentions = (head_attentions * tail_attentions).mean(dim=0)

    # normalize in order to have a distribution over sequence
    head_tail_attentions /= (
        head_tail_attentions.sum(dim=0, keepdim=True)
        + torch.finfo(head_tail_attentions.dtype).eps
    )

    # use the head_tail_attentions distribution to aggregate info from hidden_states
    return head_tail_attentions @ r1


class CV:
    """K-fold cross-validation splits of the dataset records.

    The records are split in their order (they are not shuffled).

    Args:
        keys: The record keys (``<pmid>_<sentence>_rec_<n>``).
        k: The number of folds.
    """

    def __init__(self, keys: list[str], k: int) -> None:
        self.keys = keys
        self.k = k

    def get_cv_splits(self, fold: int) -> tuple[list[str], list[str]]:
        """Return the training and test keys of ``fold``.

        The records are split into k consecutive folds; the last fold also gets the
        remaining records.
        """
        splits = []
        step = len(self.keys) // self.k
        for i in range(0, self.k * step, step):
            splits.append(self.keys[i : i + step])
        # Add the remaining keys in the last fold
        splits[-1] += self.keys[self.k * step :]
        # k-fold CV
        train_keys = []
        test_keys = []
        for i, s in enumerate(splits):
            if i == fold:
                test_keys += s
            else:
                train_keys += s
        return train_keys, test_keys

    def get_unique_sentences(self) -> list[str]:
        """Return the sentence ids (``<pmid>_<sentence>``) of the records, in order."""
        sentences = []
        for k in self.keys:
            if "_".join(k.split("_")[:2]) not in sentences:
                sentences.append("_".join(k.split("_")[:2]))
        return sentences

    def get_cv_splits_sentence_wise(self, fold: int) -> tuple[list[str], list[str]]:
        """Return the training and test keys of ``fold``, keeping sentences whole.

        All the records of a sentence are in the same fold.
        """
        sentences = self.get_unique_sentences()
        splits = []
        step = len(sentences) // self.k
        for i in range(0, self.k * step, step):
            splits.append(sentences[i : i + step])
        # Add the remaining sentences in the last fold
        splits[-1] += sentences[self.k * step :]
        splits_keys = []
        for i, s in enumerate(splits):
            temp_s = []
            for sent in s:
                for k in self.keys:
                    if "_".join(k.split("_")[:2]) == sent:
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


class SaveResults:
    """A results file, recreated with ``header`` as its first line.

    Args:
        filename: The path of the file.
        header: An optional first line.
    """

    def __init__(self, filename: str, header: str | None = None) -> None:
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)

        if header is not None:
            with open(filename, "w") as out:
                print(header, file=out)

    def save(self, info: str) -> None:
        """Append a line to the file."""
        with open(self.filename, "a") as out:
            print(info, file=out)


def set_seed(seed: int) -> None:
    """Seed the Python, NumPy and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
