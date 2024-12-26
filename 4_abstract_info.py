import argparse
from utils.pubmed import Abstract
from utils.utils import find_json_files, read_json, save_json, create_new_folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the date of extraction")
    parser.add_argument("--input_path", default="output/abstracts/", type=str,
                        required=False, help="the path that contains the abstracts")


    args = parser.parse_args()

    output_path = args.input_path + args.date + '/plots/'
    create_new_folder(output_path)

    files = find_json_files(args.input_path + args.date + '/')

    for f in files:
        file_name = f.split('/')[-1]
        if 'checked_pmids' in file_name:
            continue
        data = read_json(f)
        abstracts = Abstract(data, file_name, output_path)
        print('Number of extracted abstracts:')
        print('{}: {}' .format(file_name.split('.')[0], abstracts.number_of_abstracts()))
        print('####################')
        # Plots
        abstracts.plot_bar_chart_per_year()
        abstracts.plot_per_year()