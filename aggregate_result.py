import argparse
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


def output_name_arg(value):
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError('--output-name must not be empty.')
    if os.path.basename(value) != value:
        raise argparse.ArgumentTypeError('--output-name must be a file name, not a path.')
    return value


def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate experiment results into TSV summary files.')
    parser.add_argument(
        '--output-name',
        type=output_name_arg,
        default='experiment_results.tsv',
        help='Base file name for aggregated metric files. '
             'For example, "custom.tsv" creates "custom-dev.tsv" and "custom-test.tsv".'
    )
    return parser.parse_args()


def split_output_name(output_name):
    stem, ext = os.path.splitext(output_name)
    if ext == '':
        return output_name, '.tsv'
    return stem, ext


def build_split_output_paths(result_dir, split_name, model_name, timestamp, output_name):
    output_stem, output_ext = split_output_name(output_name)
    legacy_output = os.path.join(result_dir, f'{output_stem}-{split_name}{output_ext}')
    timestamped_output = os.path.join(result_dir, f'{output_stem}-{split_name}-{model_name}-{timestamp}{output_ext}')
    return legacy_output, timestamped_output


def load_criteria_list(result_dir, split_name):
    criteria_list = []
    split_suffix = '-' + split_name
    if not os.path.exists(result_dir):
        return criteria_list

    for result_file in os.listdir(result_dir):
        if result_file[0] == '#' and result_file.endswith(split_suffix):
            with open(os.path.join(result_dir, result_file), 'r', encoding='utf-8') as result_f:
                line = result_f.read()
                if len(line.strip()) != 0:
                    run_index, auc, mrr, ndcg5, ndcg10 = line.strip().split('\t')
                    criteria_list.append(Criteria(int(run_index[1:]), float(auc), float(mrr), float(ndcg5), float(ndcg10)))

    criteria_list.sort()
    return criteria_list


def aggregate_model_split_result(dataset, model_name, split_name, output_name, timestamp):
    result_dir = os.path.join('results', dataset, model_name)
    criteria_list = load_criteria_list(result_dir, split_name)
    if len(criteria_list) == 0:
        return None

    legacy_output, timestamped_output = build_split_output_paths(result_dir, split_name, model_name, timestamp, output_name)
    aggregate_values = write_experiment_results_file(legacy_output, model_name, criteria_list)
    write_experiment_results_file(timestamped_output, model_name, criteria_list)
    return aggregate_values


def aggregate_model_results(dataset, model_name, output_name, timestamp=None):
    timestamp = current_timestamp() if timestamp is None else timestamp
    dev_metrics = aggregate_model_split_result(dataset, model_name, 'dev', output_name, timestamp)
    test_metrics = aggregate_model_split_result(dataset, model_name, 'test', output_name, timestamp)
    return dev_metrics, test_metrics


def aggregate_result_dir(result_dir, model_label, output_name, timestamp=None):
    timestamp = current_timestamp() if timestamp is None else timestamp
    safe_model_label = model_label.replace(os.sep, '_')
    dev_metrics = None
    test_metrics = None
    for split_name in ['dev', 'test']:
        criteria_list = load_criteria_list(result_dir, split_name)
        if len(criteria_list) == 0:
            continue
        legacy_output, timestamped_output = build_split_output_paths(result_dir, split_name, safe_model_label, timestamp, output_name)
        aggregate_values = write_experiment_results_file(legacy_output, safe_model_label, criteria_list)
        write_experiment_results_file(timestamped_output, safe_model_label, criteria_list)
        if split_name == 'dev':
            dev_metrics = aggregate_values
        else:
            test_metrics = aggregate_values
    return dev_metrics, test_metrics


def write_overall_results(dataset, overall_rows, timestamp):
    legacy_overall = 'results/%s/overall.tsv' % dataset
    timestamped_overall = 'results/%s/overall-%s.tsv' % (dataset, timestamp)
    for overall_output in [legacy_overall, timestamped_overall]:
        with open(overall_output, 'w', encoding='utf-8') as overall_f:
            for model_name, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10 in overall_rows:
                overall_f.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\n' % (model_name, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))


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

def aggregate_dev_result(output_name):
    timestamp = current_timestamp()
    for dataset in ['mind', 'mind-small', 'adressa', 'adressa2']:
        if os.path.exists('results/' + dataset):
            for sub_dir in os.listdir('results/' + dataset):
                if sub_dir in list_model_name():
                    aggregate_model_split_result(dataset, sub_dir, 'dev', output_name, timestamp)


def aggregate_overall_test_result(dataset, timestamp=None):
    timestamp = current_timestamp() if timestamp is None else timestamp
    if not os.path.exists('results/' + dataset):
        return

    overall_rows = []
    for sub_dir in os.listdir('results/' + dataset):
        if sub_dir in list_model_name():
            criteria_list = load_criteria_list(os.path.join('results', dataset, sub_dir), 'test')
            if len(criteria_list) > 0:
                mean_auc = sum(criteria.auc for criteria in criteria_list) / len(criteria_list)
                mean_mrr = sum(criteria.mrr for criteria in criteria_list) / len(criteria_list)
                mean_ndcg5 = sum(criteria.ndcg5 for criteria in criteria_list) / len(criteria_list)
                mean_ndcg10 = sum(criteria.ndcg10 for criteria in criteria_list) / len(criteria_list)
                overall_rows.append((model_dict[sub_dir] if sub_dir in model_dict else sub_dir, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))

    if len(overall_rows) > 0:
        write_overall_results(dataset, overall_rows, timestamp)

def aggregate_test_result(output_name):
    timestamp = current_timestamp()
    for dataset in ['mind', 'mind-small', 'adressa', 'adressa2']:
        if os.path.exists('results/' + dataset):
            overall_rows = []
            for sub_dir in os.listdir('results/' + dataset):
                if sub_dir in list_model_name():
                    result = aggregate_model_split_result(dataset, sub_dir, 'test', output_name, timestamp)
                    if result is not None:
                        mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10 = result
                        overall_rows.append((model_dict[sub_dir] if sub_dir in model_dict else sub_dir, mean_auc, mean_mrr, mean_ndcg5, mean_ndcg10))

            if len(overall_rows) > 0:
                write_overall_results(dataset, overall_rows, timestamp)


if __name__ == '__main__':
    args = parse_args()
    aggregate_dev_result(args.output_name)
    aggregate_test_result(args.output_name)
