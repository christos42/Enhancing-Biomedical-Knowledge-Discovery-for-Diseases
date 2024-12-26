import argparse
from utils.utils import find_json_files, read_json, save_json, create_new_folder
from utils.cooccurrence import get_cooccurrence_dict_metamap, get_cooccurrence_dict, get_cooccurrence_narrow_dict, get_unique_cuis_metamap, get_unique_cuis, get_unique_cuis_narrow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="scispacy", type=str,
                        required=True, help="supported pipelines: scispacy, metamap")
    parser.add_argument("--date", type=str,
                        required=True, help="the data retrieval date")
    parser.add_argument("--input_path", default="output/mentions_extraction/", type=str,
                        required=False, help="the path of the files with extracted mentions/entities")

    args = parser.parse_args()

    output_path = args.input_path + args.date + '/' + args.pipeline + '/' + 'cooccurrence' + '/'
    create_new_folder(output_path)

    files = find_json_files(args.input_path + args.date + '/' + args.pipeline + '/merged_entities/')

    for f in files:
        file_name = f.split('/')[-1]

        data = read_json(f)
        if args.pipeline == 'scispacy':
            freq_dict = get_cooccurrence_dict(data)
            unique_cuis = get_unique_cuis(data)
            freq_dict_narrow = get_cooccurrence_narrow_dict(data)
            unique_cuis_narrow = get_unique_cuis_narrow(data)

            save_json(freq_dict, file_name, output_path)
            save_json(unique_cuis, 'unique_cuis_' + file_name, output_path)
            save_json(freq_dict_narrow, 'narrow_' + file_name, output_path)
            save_json(unique_cuis_narrow, 'narrow_unique_cuis_' + file_name, output_path)
        elif args.pipeline == 'metamap':
            freq_dict = get_cooccurrence_dict_metamap(data)
            unique_cuis = get_unique_cuis_metamap(data)

            save_json(freq_dict, file_name, output_path)
            save_json(unique_cuis, 'unique_cuis_' + file_name, output_path)