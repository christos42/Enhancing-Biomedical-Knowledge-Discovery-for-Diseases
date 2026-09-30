from __future__ import annotations

import argparse
from typing import Any

from utils.utils import find_json_files, read_json, save_json

# The order in which the knowledge bases are tried for each (grouped) entity type:
# the concepts of the first knowledge base with at least one linked concept are sampled.
CHEMICAL_PRIORITY = [
    "rxnorm",
    "gs",
    "drugbank",
    "mesh",
    "hpo",
    "go",
    "ncbi",
    "snomed",
    "umls",
]
GENE_OR_PROTEIN_PRIORITY = [
    "go",
    "hpo",
    "rxnorm",
    "gs",
    "drugbank",
    "mesh",
    "ncbi",
    "snomed",
    "umls",
]
DISEASE_PRIORITY = [
    "hpo",
    "mesh",
    "rxnorm",
    "gs",
    "drugbank",
    "go",
    "ncbi",
    "snomed",
    "umls",
]
DEFAULT_PRIORITY = [
    "hpo",
    "go",
    "rxnorm",
    "gs",
    "drugbank",
    "mesh",
    "ncbi",
    "snomed",
    "umls",
]
LINKER_PRIORITY = {
    "CHEMICAL": CHEMICAL_PRIORITY,
    "GENE_OR_PROTEIN": GENE_OR_PROTEIN_PRIORITY,
    "DNA": GENE_OR_PROTEIN_PRIORITY,
    "RNA": GENE_OR_PROTEIN_PRIORITY,
    "SO": GENE_OR_PROTEIN_PRIORITY,
    "DISEASE": DISEASE_PRIORITY,
    "PATHOLOGICAL_FORMATION": DISEASE_PRIORITY,
}


def sampling_linking_codes_strategy(data_merged: dict[str, Any]) -> dict[str, Any]:
    data_merged_upd = data_merged.copy()
    for k1 in data_merged_upd:
        for k2 in data_merged_upd[k1]:
            sampled_linked_ent_list = []
            for i, ent in enumerate(data_merged_upd[k1][k2]["entities"]):
                sampled_linked_ent = []
                linked_ent = data_merged_upd[k1][k2]["linked_entities"][i]
                sampled_linked_ent_sub: dict[str, list[Any]] = {
                    "cui": [],
                    "name": [],
                    "alias": [],
                    "tui": [],
                    "description": [],
                    "probability": [],
                    "linker": [],
                }
                for tag in list(dict.fromkeys(ent["grouped_type"])):
                    for linker in LINKER_PRIORITY.get(tag, DEFAULT_PRIORITY):
                        if len(linked_ent[linker]["cui"]) > 0:
                            dict_to_add = linked_ent[linker].copy()
                            dict_to_add["linker"] = linker
                            if dict_to_add not in sampled_linked_ent:
                                sampled_linked_ent.append(dict_to_add)
                            break
                    else:
                        # No knowledge base linked the entity
                        sampled_linked_ent.append({})

                for ent_sampled in sampled_linked_ent:
                    if not ent_sampled:
                        sampled_linked_ent_sub["linker"].append("")
                        continue
                    sampled_linked_ent_sub["cui"].extend(ent_sampled["cui"])
                    sampled_linked_ent_sub["name"].extend(ent_sampled["name"])
                    sampled_linked_ent_sub["alias"].extend(ent_sampled["alias"])
                    sampled_linked_ent_sub["tui"].extend(ent_sampled["tui"])
                    sampled_linked_ent_sub["description"].extend(
                        ent_sampled["description"]
                    )
                    sampled_linked_ent_sub["probability"].extend(
                        ent_sampled["probability"]
                    )
                    sampled_linked_ent_sub["linker"].append(ent_sampled["linker"])

                sampled_linked_ent_list.append(sampled_linked_ent_sub)
            data_merged_upd[k1][k2]["sampled_linked_entities"] = sampled_linked_ent_list

    return data_merged_upd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", type=str, required=True, help="the data retrieval date"
    )
    parser.add_argument(
        "--input_path",
        default="output/mentions_extraction/",
        type=str,
        required=False,
        help="the path of the files with extracted mentions/entities",
    )

    args = parser.parse_args()

    files = find_json_files(
        args.input_path + args.date + "/scispacy/merged_entities" + "/"
    )

    for f in files:
        file_name = f.split("/")[-1]

        data = read_json(f)
        data_upd = sampling_linking_codes_strategy(data)

        save_json(
            data_upd,
            file_name,
            args.input_path + args.date + "/scispacy/merged_entities" + "/",
        )
