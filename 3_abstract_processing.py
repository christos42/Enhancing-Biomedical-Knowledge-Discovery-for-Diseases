import argparse
from nltk.tokenize import sent_tokenize
from utils.utils import find_json_files, read_json, save_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the date of extraction")
    parser.add_argument("--input_path", default="output/abstracts/", type=str,
                        required=False, help="the path that contains the abstracts")


    args = parser.parse_args()

    files = find_json_files(args.input_path + args.date + '/')

    for f in files:
        counter = 0
        file_name = f.split('/')[-1]
        if 'checked_pmids' in file_name:
            continue
        data = read_json(f)
        for id_ in data:
            counter += 1
            if 'abstract_tokenized' in list(data[id_].keys()):
                continue
            data[id_]['abstract_tokenized'] = sent_tokenize(data[id_]['abstract'])
            data[id_]['sentence_ids'] = []
            for c in range(len(data[id_]['abstract_tokenized'])):
                data[id_]['sentence_ids'].append(id_ + '_' + str(c + 1))

            if counter % 10000 == 0:
                print('{} abstracts have been processed.' .format(counter))
                print('_')

        save_json(data, file_name, args.input_path + args.date + '/')