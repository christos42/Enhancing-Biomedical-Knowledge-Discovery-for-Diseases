"""Helpers to merge the SciSpacy NER and linking outputs (steps 5 to 7)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from spacy import displacy

if TYPE_CHECKING:
    from spacy.language import Language


def display_ner(nlp: Language, sent: str) -> None:
    """Serve a displaCy visualization of the entities found in ``sent``."""
    doc = nlp(sent)
    displacy.serve(doc, style="ent")


def find_similarity(nlp: Language, en1: str, en2: str) -> float:
    """Return the word-vector similarity of two texts."""
    # word2vec based
    doc1 = nlp(en1)
    doc2 = nlp(en2)
    return doc1.similarity(doc2)


def union_lists(l1: list[Any], l2: list[Any]) -> list[Any]:
    """Return the items of ``l1`` and then of ``l2``, without duplicates."""
    union = []
    for it in l1:
        if it not in union:
            union.append(it)
    for it in l2:
        if it not in union:
            union.append(it)

    return union


def union_lists_pairs(
    l1: list[Any], l2: list[Any], l3: list[Any], l4: list[Any]
) -> tuple[list[Any], list[Any]]:
    """Unite ``l1`` and ``l2``, carrying along the aligned items of ``l3`` and ``l4``.

    Returns:
        The union of ``l1`` and ``l2``, and for each of its items the item at the same
        index of ``l3`` (items from ``l1``) or ``l4`` (items from ``l2``).
    """
    l_un_1, l_un_2 = [], []
    for i, it in enumerate(l1):
        if it not in l_un_1:
            l_un_1.append(it)
            l_un_2.append(l3[i])
    for i, it in enumerate(l2):
        if it not in l_un_1:
            l_un_1.append(it)
            l_un_2.append(l4[i])

    return l_un_1, l_un_2


def merge_entity_pos_tags_dicts(
    dict1: dict[str, Any], dict2: dict[str, Any]
) -> dict[str, Any]:
    """Combine the outputs of two SciSpacy NER models for the same sentences.

    The entities (with their linked concepts) and the POS tags are united; the tokenized
    sentences are kept per model.
    """
    dict_: dict[str, Any] = {}
    for k1 in dict1:
        dict_[k1] = {}
        for k2 in dict1[k1]:
            dict_[k1][k2] = {}
            ent_l_1 = dict1[k1][k2]["entities"]
            linked_ent_l_1 = dict1[k1][k2]["linked_entities"]
            ent_l_2 = dict2[k1][k2]["entities"]
            linked_ent_l_2 = dict2[k1][k2]["linked_entities"]
            l_un_1, l_un_2 = union_lists_pairs(
                ent_l_1, ent_l_2, linked_ent_l_1, linked_ent_l_2
            )
            dict_[k1][k2]["entities"] = l_un_1
            dict_[k1][k2]["linked_entities"] = l_un_2

            pos_l_1 = dict1[k1][k2]["POS"]
            pos_l_2 = dict2[k1][k2]["POS"]
            dict_[k1][k2]["POS"] = union_lists(pos_l_1, pos_l_2)

            tokenized_sentence_dict = {}
            for tok1 in dict1[k1][k2]["tokenized_sentence"]:
                tokenized_sentence_dict[tok1] = dict1[k1][k2]["tokenized_sentence"][
                    tok1
                ]
            for tok2 in dict2[k1][k2]["tokenized_sentence"]:
                tokenized_sentence_dict[tok2] = dict2[k1][k2]["tokenized_sentence"][
                    tok2
                ]

            dict_[k1][k2]["tokenized_sentence"] = tokenized_sentence_dict

    return dict_


def merge_linkers_scispacy(
    d_umls: dict[str, Any],
    d_mesh: dict[str, Any],
    d_rxnorm: dict[str, Any],
    d_go: dict[str, Any],
    d_hpo: dict[str, Any],
    d_drugbank: dict[str, Any],
    d_gs: dict[str, Any],
    d_ncbi: dict[str, Any],
    d_snomed: dict[str, Any],
) -> dict[str, Any]:
    """Combine the step 5 outputs of the nine linkers.

    The entities are the same in every output, since the same NER models produced them.
    For each entity, the linked concepts of every knowledge base are gathered under the
    knowledge base's name.
    """
    d_merged: dict[str, Any] = {}
    for k1 in d_umls:
        d_merged[k1] = {}
        for k2 in d_umls[k1]:
            linked_entities = []
            for en1, en2, en3, en4, en5, en6, en7, en8, en9 in zip(
                d_umls[k1][k2]["linked_entities"],
                d_mesh[k1][k2]["linked_entities"],
                d_rxnorm[k1][k2]["linked_entities"],
                d_go[k1][k2]["linked_entities"],
                d_hpo[k1][k2]["linked_entities"],
                d_drugbank[k1][k2]["linked_entities"],
                d_gs[k1][k2]["linked_entities"],
                d_ncbi[k1][k2]["linked_entities"],
                d_snomed[k1][k2]["linked_entities"],
            ):
                linked_entities.append(
                    {
                        "umls": en1["umls"],
                        "mesh": en2["mesh"],
                        "rxnorm": en3["rxnorm"],
                        "go": en4["go"],
                        "hpo": en5["hpo"],
                        "drugbank": en6["drugbank"],
                        "gs": en7["gs"],
                        "ncbi": en8["ncbi"],
                        "snomed": en9["snomed"],
                    }
                )
            d_merged[k1][k2] = {
                "entities": d_umls[k1][k2]["entities"],
                "linked_entities": linked_entities,
                "POS": d_umls[k1][k2]["POS"],
                "tokenized_sentence": d_umls[k1][k2]["tokenized_sentence"],
            }

    return d_merged


def get_grouped_ne_tag_scispacy(tag: str) -> str:
    """Map a SciSpacy entity label to its group (e.g. SIMPLE_CHEMICAL to CHEMICAL).

    Labels outside the groups are returned unchanged.
    """
    tag_grouping = {
        "CHEMICAL": ["CHEBI", "CHEMICAL", "SIMPLE_CHEMICAL"],
        "CELL": ["CL", "CELL_TYPE", "CELL_LINE", "CELL"],
        "ORGANISM": ["ORGANISM", "TAXON"],
        "GENE_OR_PROTEIN": [
            "GGP",
            "GO",
            "PROTEIN",
            "GENE_OR_GENE_PRODUCT",
            "AMINO_ACID",
            "SO",
        ],
    }

    for k in tag_grouping:
        if tag in tag_grouping[k]:
            return k

    return tag


def merge_same_entities_scispacy(data: dict[str, Any]) -> dict[str, Any]:
    """Merge the entities that several NER models found at the same span.

    Returns:
        The data with, for each sentence, one entity per (text, start, end), listing the
        types, grouped types and models that found it.
    """
    data_entity_merging: dict[str, Any] = {}
    for k1 in data:
        data_entity_merging[k1] = {}
        for k2 in data[k1]:
            ent_dict = {}
            ent_list_triplets, linked_ent_list = [], []
            for i, en in enumerate(data[k1][k2]["entities"]):
                if (en[0], en[2], en[3]) not in ent_list_triplets:
                    ent_list_triplets.append((en[0], en[2], en[3]))
                    ent_dict[(en[0], en[2], en[3])] = {
                        "type": [en[1]],
                        "grouped_type": [get_grouped_ne_tag_scispacy(en[1])],
                        "pipeline": [en[4]],
                    }
                    linked_ent_list.append(data[k1][k2]["linked_entities"][i])
                else:
                    ent_dict[(en[0], en[2], en[3])]["type"].append(en[1])
                    ent_dict[(en[0], en[2], en[3])]["grouped_type"].append(
                        get_grouped_ne_tag_scispacy(en[1])
                    )
                    ent_dict[(en[0], en[2], en[3])]["pipeline"].append(en[4])
            ent_transformed = []
            for k in ent_dict:
                ent_transformed.append(
                    {
                        "name": k[0],
                        "start": k[1],
                        "end": k[2],
                        "type": ent_dict[k]["type"],
                        "grouped_type": ent_dict[k]["grouped_type"],
                        "pipeline": ent_dict[k]["pipeline"],
                    }
                )
            data_entity_merging[k1][k2] = {
                "entities": ent_transformed,
                "linked_entities": linked_ent_list,
                "POS": data[k1][k2]["POS"],
                "tokenized_sentence": data[k1][k2]["tokenized_sentence"],
            }
    return data_entity_merging
