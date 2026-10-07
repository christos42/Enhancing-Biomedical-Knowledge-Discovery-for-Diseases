"""Clean-up of the MetaMap Lite concepts: filtering, expansion, merging and overlaps.

Positions are MetaMap Lite ``start/length`` strings whose start is one character later
than in the sentence (pymetamap writes each sentence with a leading quote), hence the
``- 1`` when converting them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


def get_entities(d: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return the concepts of a sentence with a score of at least 0.4, by position.

    A concept found at several positions gets an entry per position; when several
    concepts share a position, the first is kept.

    Args:
        d: The MetaMap Lite concepts of the sentence (a CSV of step 5).
    """
    entities = {}
    for r in d.itertuples():
        if r.score < 0.4:
            continue
        pos = r.pos_info.split(";")
        for p in pos:
            if p not in entities:
                entities[p] = {
                    "preferred_name": r.preferred_name,
                    "cui": r.cui,
                    "semantic_type": r.semtypes,
                    "position": p,
                    "score": r.score,
                    "trigger": r.trigger,
                }

    return entities


def get_chunk(pos: str) -> list[int]:
    """Return the ``[start, end)`` character range of a MetaMap position."""
    start = int(pos.split("/")[0]) - 1
    stop = start + int(pos.split("/")[1])
    return [start, stop]


def merge_sequent_entities(
    en1: dict[str, Any], en2: dict[str, Any], chunk1: list[int], chunk2: list[int]
) -> dict[str, Any]:
    """Merge two entities into one that spans both.

    The names are joined with ``||``; the CUIs, semantic types and triggers are combined
    without duplicates.

    Args:
        en1: The entity that starts first.
        en2: The other entity.
        chunk1: The character range of ``en1``.
        chunk2: The character range of ``en2``.

    Returns:
        The merged entity, whose position covers both ranges.
    """
    m_ent = {
        "preferred_name": en1["preferred_name"] + "||" + en2["preferred_name"],
        "cui": "||".join(
            list(dict.fromkeys((en1["cui"] + "||" + en2["cui"]).split("||")))
        ),
        "semantic_type": list(
            dict.fromkeys(en1["semantic_type"] + en2["semantic_type"])
        ),
        "position": str(chunk1[0] + 1)
        + "/"
        + str(max(chunk1[1], chunk2[1]) - chunk1[0]),
        "score": [en1["score"], en2["score"]],
        "trigger": "||".join(
            list(dict.fromkeys((en1["trigger"] + "||" + en2["trigger"]).split("||")))
        ),
        "mapped_semantic_type": list(
            dict.fromkeys(en1["mapped_semantic_type"] + en2["mapped_semantic_type"])
        ),
    }

    return m_ent


def detect_overlaps(
    positions: list[str], d_: dict[str, dict[str, Any]]
) -> list[list[int]]:
    """Group the overlapping entities that have the same CUI.

    Args:
        positions: The positions of the entities (the keys of ``d_``).
        d_: Position -> entity.

    Returns:
        The groups, as indices into ``positions``.
    """
    overlaps: list[list[int]] = []
    for i1, p1 in enumerate(positions):
        for i2, p2 in enumerate(positions):
            if i1 == i2:
                continue
            else:
                p1_start = int(p1.split("/")[0]) - 1
                p1_stop = p1_start + int(p1.split("/")[1])
                p2_start = int(p2.split("/")[0]) - 1
                if (p1_start <= p2_start) and (p2_start <= p1_stop):
                    cui1 = d_[p1]["cui"]
                    cui2 = d_[p2]["cui"]
                    if cui1 == cui2:
                        flag = 0
                        for i3, o in enumerate(overlaps):
                            if i1 in o:
                                flag = 1
                                overlaps[i3].append(i2)
                            elif i2 in o:
                                flag = 1
                                overlaps[i3].insert(o.index(i2), i1)
                        if flag == 0:
                            overlaps.append([i1, i2])

    return overlaps


def resolve_overlaps(
    positions: list[str], d_: dict[str, dict[str, Any]], overlaps: list[list[int]]
) -> list[str]:
    """Choose which entity of each group of overlapping entities to drop.

    Of the first two entities of a group, the one with the lower score is dropped; on a
    tie, the shorter one.

    Returns:
        The positions of the entities to remove.
    """
    keys_to_remove = []
    for o in overlaps:
        p1 = positions[o[0]]
        p2 = positions[o[1]]
        score1 = d_[p1]["score"]
        score2 = d_[p2]["score"]
        try:
            if score1 > score2:
                keys_to_remove.append(p2)
            elif score1 < score2:
                keys_to_remove.append(p1)
            else:
                p1_start = int(p1.split("/")[0]) - 1
                p1_stop = p1_start + int(p1.split("/")[1])
                p2_start = int(p2.split("/")[0]) - 1
                p2_stop = p2_start + int(p2.split("/")[1])
                len1 = p1_stop - p1_start
                len2 = p2_stop - p2_start
                if len1 > len2:
                    keys_to_remove.append(p2)
                else:
                    keys_to_remove.append(p1)
        except Exception:
            if type(score1) is list:
                s1 = score1[0]
            else:
                s1 = score1
            if type(score2) is list:
                s2 = score2[0]
            else:
                s2 = score2
            if s1 > s2:
                keys_to_remove.append(p2)
            elif s1 < s2:
                keys_to_remove.append(p1)
            else:
                p1_start = int(p1.split("/")[0]) - 1
                p1_stop = p1_start + int(p1.split("/")[1])
                p2_start = int(p2.split("/")[0]) - 1
                p2_stop = p2_start + int(p2.split("/")[1])
                len1 = p1_stop - p1_start
                len2 = p2_stop - p2_start
                if len1 > len2:
                    keys_to_remove.append(p2)
                else:
                    keys_to_remove.append(p1)

    return keys_to_remove


def resolve_overlaps_with_expansion(
    positions: list[str], d_: dict[str, dict[str, Any]]
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    """Merge every pair of overlapping entities, whatever their CUIs.

    Returns:
        The position pairs of the entities that were merged (to remove), and the merged
        entities.
    """
    merged_entities = []
    keys_to_remove = []
    for i1, p1 in enumerate(positions):
        for i2, p2 in enumerate(positions):
            if i1 == i2:
                continue
            else:
                p1_start = int(p1.split("/")[0]) - 1
                p1_stop = p1_start + int(p1.split("/")[1])
                p2_start = int(p2.split("/")[0]) - 1
                if (p1_start <= p2_start) and (p2_start <= p1_stop):
                    ent1 = d_[p1]
                    ent2 = d_[p2]
                    chunk1 = get_chunk(p1)
                    chunk2 = get_chunk(p2)
                    # Merge the entities
                    f_m_ent = merge_sequent_entities(ent1, ent2, chunk1, chunk2)
                    merged_entities.append(f_m_ent)
                    keys_to_remove.append([p1, p2])

    return keys_to_remove, merged_entities


def check_expansion(position: str, sentence: str) -> tuple[int, str]:
    """Expand an entity to the whole word(s) it is part of.

    The entity is extended backwards to the previous space and forwards to the next
    space (or to the full stop that ends the sentence).

    Returns:
        1 if the position changed (else 0), and the new position.
    """
    p_start, p_stop = get_chunk(position)
    # Index of the last character of the expanded entity (the end of the sentence if no
    # boundary follows)
    new_p_stop = len(sentence) - 1
    for index in range(p_stop, len(sentence)):
        # if (sentence[index] in [' ', '(', ')', '<', '>']) or (
        #         sentence[index] == '.' and index == len(sentence) - 1):
        # if (sentence[index] in [' ', ',']) or (
        #         sentence[index] == '.' and index == len(sentence) - 1):
        if (sentence[index] in [" "]) or (
            sentence[index] == "." and index == len(sentence) - 1
        ):
            new_p_stop = index - 1
            break

    new_p_start = p_start
    index = p_start - 1
    while index >= 0:
        # if (sentence[index] in [' ', '(', ')', '<', '>']):
        if sentence[index] in [" "]:
            break
        new_p_start = index
        index -= 1

    if new_p_stop == p_stop - 1 and new_p_start == p_start:
        update = 0
    else:
        update = 1

    new_position = str(new_p_start + 1) + "/" + str(new_p_stop - new_p_start + 1)
    return update, new_position


def expand_entities(
    entities: dict[str, dict[str, Any]], sentence: str
) -> dict[str, dict[str, Any]]:
    """Expand every entity of a sentence (see check_expansion).

    The entities are keyed by their new positions.
    """
    updated_dict = {}
    for k in entities:
        update, new_position = check_expansion(k, sentence)
        if update == 1:
            updated_dict[new_position] = entities[k].copy()
            updated_dict[new_position]["position"] = new_position
        else:
            updated_dict[k] = entities[k].copy()
    return updated_dict
