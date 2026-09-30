def get_cooccurrence_dict_metamap(data):
    freq_pairs = {}
    for k1 in data:
        for i, ent1_ in enumerate(list(data[k1].keys())):
            ent1 = data[k1][ent1_]
            cui1 = ent1["cui"]
            cui1 = cui1.replace("||", "_")
            for j, ent2_ in enumerate(list(data[k1].keys())[i + 1 :]):
                ent2 = data[k1][ent2_]
                cui2 = ent2["cui"]
                cui2 = cui2.replace("||", "_")
                if cui1 == cui2:
                    continue
                if (
                    cui1 + "_" + cui2 not in freq_pairs.keys()
                    and cui2 + "_" + cui1 not in freq_pairs.keys()
                ):
                    freq_pairs[cui1 + "_" + cui2] = {
                        "frequency": 1,
                        "cui_pair": (cui1, cui2),
                        "textual_pair": (
                            ent1["preferred_name"],
                            ent2["preferred_name"],
                        ),
                        "semantic_types": (
                            ent1["mapped_semantic_type"],
                            ent2["mapped_semantic_type"],
                        ),
                        "sentence_ids": [k1],
                    }
                elif cui1 + "_" + cui2 in freq_pairs.keys():
                    freq_pairs[cui1 + "_" + cui2]["frequency"] += 1
                    freq_pairs[cui1 + "_" + cui2]["sentence_ids"].append(k1)
                elif cui2 + "_" + cui1 in freq_pairs.keys():
                    freq_pairs[cui2 + "_" + cui1]["frequency"] += 1
                    freq_pairs[cui2 + "_" + cui1]["sentence_ids"].append(k1)

    freq_pairs_sorted = dict(
        sorted(freq_pairs.items(), key=lambda item: item[1]["frequency"], reverse=True)
    )

    return freq_pairs_sorted


def get_cooccurrence_dict(data):
    freq_pairs = {}
    for k1 in data:
        for k2 in data[k1]:
            for i, ent1 in enumerate(data[k1][k2]["sampled_linked_entities"]):
                for j, ent2 in enumerate(
                    data[k1][k2]["sampled_linked_entities"][i + 1 :]
                ):
                    for cui1 in dict.fromkeys(ent1["cui"]):
                        i1 = ent1["cui"].index(cui1)
                        for cui2 in dict.fromkeys(ent2["cui"]):
                            i2 = ent2["cui"].index(cui2)
                            if cui1 == cui2:
                                continue
                            if (
                                cui1 + "_" + cui2 not in freq_pairs.keys()
                                and cui2 + "_" + cui1 not in freq_pairs.keys()
                            ):
                                freq_pairs[cui1 + "_" + cui2] = {
                                    "frequency": 1,
                                    "cui_pair": (cui1, cui2),
                                    "textual_pair": (
                                        ent1["name"][i1],
                                        ent2["name"][i2],
                                    ),
                                    "alias": (ent1["alias"][i1], ent2["alias"][i2]),
                                    "descriptions": (
                                        ent1["description"][i1],
                                        ent2["description"][i2],
                                    ),
                                    "sentence_ids": [k2],
                                }
                            elif cui1 + "_" + cui2 in freq_pairs.keys():
                                freq_pairs[cui1 + "_" + cui2]["frequency"] += 1
                                freq_pairs[cui1 + "_" + cui2]["sentence_ids"].append(k2)
                            elif cui2 + "_" + cui1 in freq_pairs.keys():
                                freq_pairs[cui2 + "_" + cui1]["frequency"] += 1
                                freq_pairs[cui2 + "_" + cui1]["sentence_ids"].append(k2)

    freq_pairs_sorted = dict(
        sorted(freq_pairs.items(), key=lambda item: item[1]["frequency"], reverse=True)
    )

    return freq_pairs_sorted


def get_cooccurrence_narrow_dict(data):
    freq_pairs = {}
    for k1 in data:
        for k2 in data[k1]:
            for i, ent1 in enumerate(data[k1][k2]["sampled_linked_entities"]):
                for j, ent2 in enumerate(
                    data[k1][k2]["sampled_linked_entities"][i + 1 :]
                ):
                    if len(ent1["cui"]) > 0 and len(ent2["cui"]) > 0:
                        cui1 = ent1["cui"][0]
                        cui2 = ent2["cui"][0]
                        if cui1 == cui2:
                            continue
                        if (
                            cui1 + "_" + cui2 not in freq_pairs.keys()
                            and cui2 + "_" + cui1 not in freq_pairs.keys()
                        ):
                            freq_pairs[cui1 + "_" + cui2] = {
                                "frequency": 1,
                                "cui_pair": (cui1, cui2),
                                "textual_pair": (ent1["name"][0], ent2["name"][0]),
                                "alias": (ent1["alias"][0], ent2["alias"][0]),
                                "descriptions": (
                                    ent1["description"][0],
                                    ent2["description"][0],
                                ),
                                "sentence_ids": [k2],
                            }
                        elif cui1 + "_" + cui2 in freq_pairs.keys():
                            freq_pairs[cui1 + "_" + cui2]["frequency"] += 1
                            freq_pairs[cui1 + "_" + cui2]["sentence_ids"].append(k2)
                        elif cui2 + "_" + cui1 in freq_pairs.keys():
                            freq_pairs[cui2 + "_" + cui1]["frequency"] += 1
                            freq_pairs[cui2 + "_" + cui1]["sentence_ids"].append(k2)

    freq_pairs_sorted = dict(
        sorted(freq_pairs.items(), key=lambda item: item[1]["frequency"], reverse=True)
    )

    return freq_pairs_sorted


def get_unique_cuis_metamap(data):
    # Used as an insertion-ordered set: constant-time membership, unlike a list
    unique_cuis = {}
    for k1 in data:
        for k2 in data[k1].keys():
            ent = data[k1][k2]
            cuis = ent["cui"].split("||")
            for cui in cuis:
                unique_cuis[cui] = None

    return list(unique_cuis)


def get_unique_cuis(data):
    # Used as an insertion-ordered set: constant-time membership, unlike a list
    unique_cuis = {}
    for k1 in data:
        for k2 in data[k1]:
            for ent in data[k1][k2]["sampled_linked_entities"]:
                for cui in ent["cui"]:
                    unique_cuis[cui] = None

    return list(unique_cuis)


def get_unique_cuis_narrow(data):
    # Used as an insertion-ordered set: constant-time membership, unlike a list
    unique_cuis = {}
    for k1 in data:
        for k2 in data[k1]:
            for ent in data[k1][k2]["sampled_linked_entities"]:
                if len(ent["cui"]) == 0:
                    continue
                unique_cuis[ent["cui"][0]] = None

    return list(unique_cuis)
