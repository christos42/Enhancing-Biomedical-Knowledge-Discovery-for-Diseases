import argparse
from utils.pubmed import PubMed
from datetime import date
from utils.utils import save_json, create_new_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", default="", type=str,
                        required=True, help="the query term for the search, e.g. dementia")
    parser.add_argument("--output_path", default="output/pmid/", type=str,
                        required=False, help="the output path")

    args = parser.parse_args()

    # Current date
    today = date.today()
    current_date = today.strftime("%d_%m_%y")

    # Create new folder if needed
    create_new_folder(args.output_path + current_date)

    p = PubMed(args.query.split(','))
    print('Query: {}'.format(args.query))
    count = p.total_number_of_docs()
    file_name = "_".join(args.query.split(',')) + '.json'
    file_name = file_name.replace(' ', '')


    if int(count) <= 9999:
        s = p.search('', '')
        save_json(s['IdList'], file_name, args.output_path + current_date + '/')
        print('Number of PMIDs: {}'.format(len(s['IdList'])))
    else:
        ids, _ = p.retrieve_all_ids()
        save_json(ids, new_file_name, args.output_path + current_date + '/')
        print('Number of PMIDs: {}'.format(len(ids)))
    print('#############################')