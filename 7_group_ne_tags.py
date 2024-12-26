import argparse
from utils.utils import find_json_files, read_json, save_json, create_new_folder
from utils.ner_utils import merge_same_entities_scispacy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the data retrieval date")
    parser.add_argument("--input_path", default="output/mentions_extraction/", type=str,
                        required=False, help="the path of the files with extracted mentions/entities")

    args = parser.parse_args()

    output_path = args.input_path + args.date + '/scispacy/merged_entities' + '/'
    create_new_folder(output_path)

    files = find_json_files(args.input_path + args.date + '/scispacy/merged_linkers' + '/')

    for f in files:
        file_name = f.split('/')[-1]
        data = read_json(f)
        data_merged = merge_same_entities_scispacy(data)

        save_json(data_merged, file_name, output_path)