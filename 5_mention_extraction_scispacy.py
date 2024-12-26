import argparse
from utils.utils import find_json_files, read_json, save_json, create_new_folder
from utils.mentions_extractor import MentionsExtractorSciSpacy
from utils.ner_utils import merge_entity_pos_tags_dicts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the date of extraction")
    parser.add_argument("--linker", default="umls", type=str, required=False,
                        help="supported linkers: umls, mesh, rxnorm, go, hpo, drugbank, gs, ncbi, snomed")
    parser.add_argument("--input_path", default="output/abstracts/", type=str,
                        required=False, help="the path that contains the abstracts")
    parser.add_argument("--output_path", default="output/mentions_extraction/",
                        type=str, required=False, help="the output path")

    args = parser.parse_args()

    output_path = args.output_path + args.date + '/scispacy/' + '/linking/' + args.linker + '/'
    create_new_folder(output_path)

    files = find_json_files(args.input_path + args.date + '/')

    scispacy_craft = MentionsExtractorSciSpacy('craft', args.linker)
    scispacy_bc5cdr = MentionsExtractorSciSpacy('bc5cdr', args.linker)
    scispacy_jnlpba = MentionsExtractorSciSpacy('jnlpba', args.linker)
    scispacy_bionlp13cg = MentionsExtractorSciSpacy('bionlp13cg', args.linker)

    for f in files:
        file_name = f.split('/')[-1]
        data = read_json(f)
        info_craft = scispacy_craft.extract_entities_pos_tags(data)
        info_bc5cdr = scispacy_bc5cdr.extract_entities_pos_tags(data)
        info_jnlpba= scispacy_jnlpba.extract_entities_pos_tags(data)
        info_bionlp13cg = scispacy_bionlp13cg.extract_entities_pos_tags(data)

        info1 = merge_entity_pos_tags_dicts(info_craft, info_bc5cdr)
        info2 = merge_entity_pos_tags_dicts(info1, info_jnlpba)
        info3 = merge_entity_pos_tags_dicts(info2, info_bionlp13cg)

        save_json(info3, file_name, output_path)