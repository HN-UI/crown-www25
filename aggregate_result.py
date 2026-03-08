import os
import math
from datetime import datetime


model_dict = {
    'PNE-PUE': 'NPA',
    'CNN-LSTUR': 'LSTUR',
    'NAML-ATT': 'NAML',
    'MHSA-MHSA': 'NRMS',
    'HDC-FIM': 'FIM',
    'CNE-SUE': 'CNE-SUE',
    'CROWN-CROWN': 'CROWN'
}


def current_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class Criteria:
    def __init__(self, run_index, auc, mrr, ndcg5, ndcg10):
        self.run_index = run_index
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10

    def __gt__(self, value):
        return self.run_index > value.run_index

    def __ge__(self, value):
        return self.run_index >= value.run_index

    def __lt__(self, value):
        return self.run_index < value.run_index

    def __le__(self, value):
        return self.run_index <= value.run_index

    def __str__(self):
        return '#%d\t%.4f\t%.4f\t%.4f\t%.4f' % (self.run_index, self.auc, self.mrr, self.ndcg5, self.ndcg10)

def aggregate_criteria(model_name, criteria_list, experiment_results_f):
    n = len(criteria_list)
    auc_values = [criteria.auc for criteria in criteria_list]
    mrr_values = [criteria.mrr for criteria in criteria_list]
    ndcg5_values = [criteria.ndcg5 for criteria in criteria_list]
    ndcg10_values = [criteria.ndcg10 for criteria in criteria_list]
    mean_auc = sum(auc_values) / n
    mean_mrr = sum(mrr_values) / n
    mean_ndcg5 = sum(ndcg5_values) / n
    mean_ndcg10 = sum(ndcg10_values) / n
    std_auc = math.sqrt(sum((x - mean_auc) ** 2 for x in auc_values) / n)
    std_mrr = math.sqrt(sum((x - mean_mrr) ** 2 for x in mrr_values) / n)
    std_ndcg5 = math.sqrt(sum((x - mean_ndcg5) ** 2 for x in ndcg5_values) / n)
    std_ndcg10 = math.sqrt(sum((x - mean_ndcg10) ** 2 for x in ndcg10_values) / n)
    experiment_results_f.write('\nAvg\t%.4f\t%.4f\t%.4f\t%.4f\n' % (mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))
    experiment_results_f.write('Std\t%.4f\t%.4f\t%.4f\t%.4f\n' % (std_auc, std_mrr, std_ndcg5, std_ndcg10))
    return mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10


def write_experiment_results_file(output_path, model_name, criteria_list):
    with open(output_path, 'w', encoding='utf-8') as experiment_results_f:
        experiment_results_f.write('exp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
        for criteria in criteria_list:
            experiment_results_f.write(str(criteria) + '\n')
        return aggregate_criteria(model_name, criteria_list, experiment_results_f)

def list_model_name():
    model_names = []
    for news_encoder in ['CNE', 'CNN', 'MHSA', 'NAML', 'PNE']:
        for user_encoder in ['SUE', 'LSTUR', 'MHSA', 'ATT', 'PUE', 'GRU']:
            model_names.append(news_encoder + '-' + user_encoder)
    return model_names

def aggregate_dev_result():
    timestamp = current_timestamp()
    for dataset in ['mind', 'mind-small', 'adressa', 'adressa2']:
        if os.path.exists('results/' + dataset):
            for sub_dir in os.listdir('results/' + dataset):
                if sub_dir in list_model_name():
                    criteria_list = []
                    for result_file in os.listdir('results/' + dataset + '/' + sub_dir):
                        if result_file[0] == '#' and result_file[-4:] == '-dev':
                            with open('results/' + dataset + '/' + sub_dir + '/' + result_file, 'r', encoding='utf-8') as result_f:
                                line = result_f.read()
                                if len(line.strip()) != 0:
                                    run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                                    criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))
                    if len(criteria_list) > 0:
                        criteria_list.sort()
                        legacy_output = 'results/' + dataset + '/' + sub_dir + '/experiment_results-dev.tsv'
                        timestamped_output = 'results/' + dataset + '/' + sub_dir + '/experiment_results-dev-' + sub_dir + '-' + timestamp + '.tsv'
                        write_experiment_results_file(legacy_output, sub_dir, criteria_list)
                        write_experiment_results_file(timestamped_output, sub_dir, criteria_list)

def aggregate_test_result():
    timestamp = current_timestamp()
    for dataset in ['mind', 'mind-small', 'adressa', 'adressa2']:
        if os.path.exists('results/' + dataset):
            overall_rows = []
            for sub_dir in os.listdir('results/' + dataset):
                if sub_dir in list_model_name():
                    criteria_list = []
                    for result_file in os.listdir('results/' + dataset + '/' + sub_dir):
                        if result_file[0] == '#' and result_file[-5:] == '-test':
                            with open('results/' + dataset + '/' + sub_dir + '/' + result_file, 'r', encoding='utf-8') as result_f:
                                line = result_f.read()
                                if len(line.strip()) != 0:
                                    run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                                    criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))
                    if len(criteria_list) > 0:
                        criteria_list.sort()
                        legacy_output = 'results/' + dataset + '/' + sub_dir + '/experiment_results-test.tsv'
                        timestamped_output = 'results/' + dataset + '/' + sub_dir + '/experiment_results-test-' + sub_dir + '-' + timestamp + '.tsv'
                        mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10 = write_experiment_results_file(legacy_output, sub_dir, criteria_list)
                        write_experiment_results_file(timestamped_output, sub_dir, criteria_list)
                        overall_rows.append((model_dict[sub_dir] if sub_dir in model_dict else sub_dir, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))

            if len(overall_rows) > 0:
                legacy_overall = 'results/%s/overall.tsv' % dataset
                timestamped_overall = 'results/%s/overall-%s.tsv' % (dataset, timestamp)
                for overall_output in [legacy_overall, timestamped_overall]:
                    with open(overall_output, 'w', encoding='utf-8') as overall_f:
                        for model_name, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10 in overall_rows:
                            overall_f.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\n' % (model_name, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))


if __name__ == '__main__':
    aggregate_dev_result()
    aggregate_test_result()
