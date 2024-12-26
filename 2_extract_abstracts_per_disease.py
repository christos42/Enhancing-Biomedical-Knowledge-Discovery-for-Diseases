import argparse
from utils.pubmed import PubMed
from datetime import date
from utils.utils import save_json, read_json, find_json_files, no_intersection_lists, create_new_folder

def get_unique_abstracts(all_abstracts):
    # Applicable if PubMedDivide class is used for abstract extraction.
    unique_abstracts = {}
    for date in all_abstracts.keys():
        for id_ in all_abstracts[date].keys():
            if id_ not in unique_abstracts.keys():
                unique_abstracts[id_] = all_abstracts[date][id_]

    return unique_abstracts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the date of extraction")
    parser.add_argument("--input_path", default="output/pmid/", type=str,
                        required=False, help="the path containing PMIDs lists")
    parser.add_argument("--output_path", default="output/abstracts/", type=str,
                        required=False, help="the output path where the abstracts are stored")

    args = parser.parse_args()

    # Create new folder if needed
    create_new_folder(args.output_path + args.date)

    pmid_files = find_json_files(args.input_path + args.date + '/')

    not_completed = []
    for f in pmid_files:
        flag = 0
        file_name = f.split('/')[-1]
        try:
            abstracts_ready = read_json(args.output_path + args.date + '/' + file_name)
            pmids_checked = read_json(args.output_path + args.date + '/' + file_name.split('.')[0] + '_checked_pmids.json')
        except:
            abstracts_ready = {}
            pmids_checked = []
        pmid_all = read_json(f)
        pmid = no_intersection_lists(pmid_all, pmids_checked)
        p = PubMed('')
        try:
            for i in range(0, len(pmid), 5000):
                if i + 5000 >= len(pmid):
                    abstracts = p.retrieve_abstracts(pmid[i:])
                    pmids_checked.extend(pmid[i:])
                else:
                    abstracts = p.retrieve_abstracts(pmid[i:i + 5000])
                    pmids_checked.extend(pmid[i:i + 5000])
                abstracts_ready.update(abstracts)
        except:
            flag = 1
            save_json(abstracts_ready, file_name, args.output_path + args.date + '/')
            save_json(pmids_checked, file_name.split('.')[0] + '_checked_pmids.json', args.output_path + args.date + '/')
            not_completed.append(file_name)
        if flag == 0:
            save_json(abstracts_ready, file_name, args.output_path + args.date + '/')
            save_json(pmids_checked, file_name.split('.')[0] + '_checked_pmids.json', args.output_path + args.date + '/')

    for f in not_completed:
        print(f)

    if len(not_completed) == 0:
        print('All abstracts have been extracted successfully!')