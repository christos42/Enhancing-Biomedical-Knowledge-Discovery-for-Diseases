import argparse
from utils.utils import find_json_files, read_json, save_json, create_new_folder
from utils.ner_utils import merge_linkers_scispacy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str,
                        required=True, help="the data retrieval date")
    parser.add_argument("--input_path", default="output/mentions_extraction/", type=str,
                        required=False, help="the path of the files with extracted mentions/entities")

    args = parser.parse_args()

    output_path = args.input_path + args.date + '/scispacy/merged_linkers/'
    create_new_folder(output_path)

    files_umls = find_json_files(args.input_path + args.date + '/scispacy/linking/umls/')
    files_mesh = find_json_files(args.input_path + args.date + '/scispacy/linking/mesh/')
    files_rxnorm = find_json_files(args.input_path + args.date + '/scispacy/linking/rxnorm/')
    files_go = find_json_files(args.input_path + args.date + '/scispacy/linking/go/')
    files_hpo = find_json_files(args.input_path + args.date + '/scispacy/linking/hpo/')
    files_drugbank = find_json_files(args.input_path + args.date + '/scispacy/linking/drugbank/')
    files_gs = find_json_files(args.input_path + args.date + '/scispacy/linking/gs/')
    files_ncbi = find_json_files(args.input_path + args.date + '/scispacy/linking/ncbi/')
    files_snomed = find_json_files(args.input_path + args.date + '/scispacy/linking/snomed/')


    for f_umls, f_mesh, f_rxnorm, f_go, f_hpo, f_drugbank, f_gs, f_ncbi, f_snomed in zip(files_umls,
                                                                                         files_mesh,
                                                                                         files_rxnorm,
                                                                                         files_go,
                                                                                         files_hpo,
                                                                                         files_drugbank,
                                                                                         files_gs,
                                                                                         files_ncbi,
                                                                                         files_snomed):
        file_name = f_umls.split('/')[-1]
        d_umls = read_json(f_umls)
        d_mesh = read_json(f_mesh)
        d_rxnorm = read_json(f_rxnorm)
        d_go = read_json(f_go)
        d_hpo = read_json(f_hpo)
        d_drugbank = read_json(f_drugbank)
        d_gs = read_json(f_gs)
        d_ncbi = read_json(f_ncbi)
        d_snomed = read_json(f_snomed)

        d_merged = merge_linkers_scispacy(d_umls, d_mesh, d_rxnorm, d_go, d_hpo, d_drugbank, d_gs, d_ncbi, d_snomed)
        save_json(d_merged, file_name, output_path)