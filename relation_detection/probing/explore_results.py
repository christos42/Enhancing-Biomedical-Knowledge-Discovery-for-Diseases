import os
import numpy as np
import json
import argparse


class Search:
    def __init__(self, folder_path, output_path):
        self.folder_path = folder_path
        self.output_path = output_path
        self.files_to_check = self.get_files()
        self.res_dict = self.build_res_dict()
        self.add_avg_metrics_cv()

    def get_files(self):
        training_info_files = []
        for root, _, files in os.walk(self.folder_path, topdown=False):
            for name in files:
                if name.endswith('txt'):
                    training_info_files.append(os.path.join(root, name))
        return training_info_files

    def build_res_dict(self):
        res_dict = {}
        for f in self.files_to_check:
            with open(f) as f_:
                lines = f_.readlines()
                last_line_splits = lines[-1].split(' ')
                try:
                    prec = last_line_splits[4]
                    rec = last_line_splits[7]
                    f1 = last_line_splits[10]
                    f_0_5 = last_line_splits[-1][:-1]
                    f_sub = f.split('/')
                    if f_sub[1] not in res_dict.keys():
                        res_dict[f_sub[1]] = {}
                    if f_sub[2] not in res_dict[f_sub[1]].keys():
                        res_dict[f_sub[1]][f_sub[2]] = {}
                    if f_sub[3] not in res_dict[f_sub[1]][f_sub[2]].keys():
                        res_dict[f_sub[1]][f_sub[2]][f_sub[3]] = {}
                    if f_sub[4] not in res_dict[f_sub[1]][f_sub[2]][f_sub[3]].keys():
                        res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]] = {}
                    if f_sub[5] not in res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]].keys():
                        res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]] = {}
                    if f_sub[6] not in res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]].keys():
                        res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]][f_sub[6]] = {}
                    if f_sub[7] not in res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]][f_sub[6]].keys():
                        res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]][f_sub[6]][f_sub[7]] = {'prec': [],
                                                                                                          'rec': [],
                                                                                                          'f1': [],
                                                                                                          'f_0_5': []}

                    res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]][f_sub[6]][f_sub[7]]['prec'].append(float(prec))
                    res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]][f_sub[6]][f_sub[7]]['rec'].append(float(rec))
                    res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]][f_sub[6]][f_sub[7]]['f1'].append(float(f1))
                    res_dict[f_sub[1]][f_sub[2]][f_sub[3]][f_sub[4]][f_sub[5]][f_sub[6]][f_sub[7]]['f_0_5'].append(float(f_0_5))
                except:
                    pass

        return res_dict

    def add_avg_metrics_cv(self):
        for k1 in self.res_dict.keys():
            for k2 in self.res_dict[k1].keys():
                for k3 in self.res_dict[k1][k2].keys():
                    for k4 in self.res_dict[k1][k2][k3].keys():
                        for k5 in self.res_dict[k1][k2][k3][k4].keys():
                            for k6 in self.res_dict[k1][k2][k3][k4][k5].keys():
                                for k7 in self.res_dict[k1][k2][k3][k4][k5][k6].keys():
                                    try:
                                        self.res_dict[k1][k2][k3][k4][k5][k6][k7]['avg_prec'] = round(
                                            np.mean(self.res_dict[k1][k2][k3][k4][k5][k6][k7]['prec']), 4)
                                        self.res_dict[k1][k2][k3][k4][k5][k6][k7]['avg_rec'] = round(
                                            np.mean(self.res_dict[k1][k2][k3][k4][k5][k6][k7]['rec']), 4)
                                        self.res_dict[k1][k2][k3][k4][k5][k6][k7]['avg_f1'] = round(
                                            np.mean(self.res_dict[k1][k2][k3][k4][k5][k6][k7]['f1']), 4)
                                        self.res_dict[k1][k2][k3][k4][k5][k6][k7]['avg_f_0_5'] = round(
                                            np.mean(self.res_dict[k1][k2][k3][k4][k5][k6][k7]['f_0_5']), 4)
                                    except:
                                        pass


    def save_res_dict(self):
        with open(self.output_path + 'overall_results.json', 'w') as outfile:
            json.dump(self.res_dict, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_models_path", default='results/', type=str, required=False,
                        help="The path of the saved models.")
    parser.add_argument("--output_path", default='results/', type=str, required=False,
                        help="The output path for storing the aggregated results.")

    args = parser.parse_args()

    obj = Search(args.saved_models_path,
                 args.output_path)
    obj.save_res_dict()