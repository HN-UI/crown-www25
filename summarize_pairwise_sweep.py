#!/usr/bin/env python3
import argparse
import math
import os
from typing import Dict, List, Optional, Tuple


DEFAULT_WEIGHTS = ['0.1', '0.2', '0.4']
METRIC_NAMES = ['AUC', 'MRR', 'nDCG@5', 'nDCG@10']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Summarize coordinated pairwise sweep results.')
    parser.add_argument('--news-encoder', default='NAML')
    parser.add_argument('--user-encoder', default='ATT')
    parser.add_argument('--dataset', default='mind')
    parser.add_argument('--mind-size', default='small')
    parser.add_argument('--pairwise-loss', default='log_sigmoid', choices=['log_sigmoid', 'margin'])
    parser.add_argument('--num-runs', type=int, default=5)
    parser.add_argument('--weights', nargs='*', default=DEFAULT_WEIGHTS)
    return parser.parse_args()


def safe_weight(weight: str) -> str:
    return weight.replace('.', 'p')


def dataset_tag(dataset: str, mind_size: str) -> str:
    if dataset == 'mind':
        return f'mind-{mind_size}'
    return dataset


def model_name(news_encoder: str, user_encoder: str) -> str:
    return f'{news_encoder}-{user_encoder}'


def output_prefix(news_encoder: str, user_encoder: str, pairwise_loss: str) -> str:
    return f'{news_encoder}_{user_encoder}_PAIRWISE_{pairwise_loss}'


def state_root(news_encoder: str, user_encoder: str, pairwise_loss: str) -> str:
    return os.path.join('sweep_state', f'{news_encoder.lower()}_{user_encoder.lower()}_pairwise_{pairwise_loss}')


def parse_raw_result_file(path: str) -> Optional[Tuple[int, List[float]]]:
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as result_f:
        line = result_f.read().strip()
    if not line:
        return None
    parts = line.split('\t')
    if len(parts) != 5 or not parts[0].startswith('#'):
        return None
    try:
        run_index = int(parts[0][1:])
        metrics = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    return run_index, metrics


def load_split_results(result_dir: str, split_name: str) -> List[Tuple[int, List[float]]]:
    results: List[Tuple[int, List[float]]] = []
    if not os.path.isdir(result_dir):
        return results
    suffix = f'-{split_name}'
    for name in os.listdir(result_dir):
        if not name.startswith('#') or not name.endswith(suffix):
            continue
        parsed = parse_raw_result_file(os.path.join(result_dir, name))
        if parsed is not None:
            results.append(parsed)
    results.sort(key=lambda item: item[0])
    return results


def mean_and_std(selected: List[Tuple[int, List[float]]]) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    if not selected:
        return [None] * len(METRIC_NAMES), [None] * len(METRIC_NAMES)

    metric_columns = list(zip(*[metrics for _, metrics in selected]))
    means = [sum(values) / len(values) for values in metric_columns]
    stds = []
    for mean, values in zip(means, metric_columns):
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        stds.append(math.sqrt(variance))
    return means, stds


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return ''
    return f'{value:.4f}'


def write_split_summary(path: str, selected: List[Tuple[int, List[float]]]) -> None:
    with open(path, 'w', encoding='utf-8') as output_f:
        output_f.write('exp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
        for run_index, metrics in selected:
            output_f.write('#%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (run_index, metrics[0], metrics[1], metrics[2], metrics[3]))
        means, stds = mean_and_std(selected)
        output_f.write('Avg\t%s\t%s\t%s\t%s\n' % tuple(format_metric(value) for value in means))
        output_f.write('Std\t%s\t%s\t%s\t%s\n' % tuple(format_metric(value) for value in stds))


def write_weight_overview(path: str, summary: Dict[str, object]) -> None:
    has_any_result = bool(summary['dev'] or summary['test'])
    if not has_any_result:
        if os.path.exists(path):
            os.remove(path)
        return

    with open(path, 'w', encoding='utf-8') as output_f:
        output_f.write('split\texp_ID\tAUC\tMRR\tnDCG@5\tnDCG@10\n')
        for split_name in ['dev', 'test']:
            selected = summary[split_name]
            means = summary[f'{split_name}_mean']
            stds = summary[f'{split_name}_std']
            for run_index, metrics in selected:
                output_f.write('%s\t#%d\t%.4f\t%.4f\t%.4f\t%.4f\n' % (split_name, run_index, metrics[0], metrics[1], metrics[2], metrics[3]))
            if selected:
                output_f.write('%s\tAvg\t%s\t%s\t%s\t%s\n' % ((split_name,) + tuple(format_metric(value) for value in means)))
                output_f.write('%s\tStd\t%s\t%s\t%s\t%s\n' % ((split_name,) + tuple(format_metric(value) for value in stds)))


def summarize_weight(result_dir: str, output_name: str, num_runs: int) -> Dict[str, object]:
    output_stem, _ = os.path.splitext(output_name)
    dev_results = load_split_results(result_dir, 'dev')
    test_results = load_split_results(result_dir, 'test')
    selected_dev = dev_results[-num_runs:]
    selected_test = test_results[-num_runs:]

    if selected_dev:
        write_split_summary(os.path.join(result_dir, f'{output_stem}-dev.tsv'), selected_dev)
    if selected_test:
        write_split_summary(os.path.join(result_dir, f'{output_stem}-test.tsv'), selected_test)

    if len(selected_dev) >= num_runs and len(selected_test) >= num_runs:
        status = 'done'
    elif selected_dev or selected_test:
        status = 'partial'
    else:
        status = 'pending'

    dev_mean, dev_std = mean_and_std(selected_dev)
    test_mean, test_std = mean_and_std(selected_test)
    return {
        'status': status,
        'dev': selected_dev,
        'test': selected_test,
        'dev_mean': dev_mean,
        'dev_std': dev_std,
        'test_mean': test_mean,
        'test_std': test_std,
    }


def build_header(num_runs: int) -> List[str]:
    header = ['weight', 'status', 'result_dir', 'dev_run_count', 'test_run_count']
    for split_name in ['dev', 'test']:
        for run_slot in range(1, num_runs + 1):
            header.append(f'{split_name}_run{run_slot}_id')
            for metric_name in METRIC_NAMES:
                header.append(f'{split_name}_run{run_slot}_{metric_name}')
        for metric_name in METRIC_NAMES:
            header.append(f'{split_name}_avg_{metric_name}')
        for metric_name in METRIC_NAMES:
            header.append(f'{split_name}_std_{metric_name}')
    return header


def build_row(weight: str, result_dir: str, summary: Dict[str, object], num_runs: int) -> List[str]:
    row = [weight, str(summary['status']), result_dir, str(len(summary['dev'])), str(len(summary['test']))]
    for split_name in ['dev', 'test']:
        selected = summary[split_name]
        for run_slot in range(num_runs):
            if run_slot < len(selected):
                run_index, metrics = selected[run_slot]
                row.append(str(run_index))
                row.extend(format_metric(value) for value in metrics)
            else:
                row.append('')
                row.extend([''] * len(METRIC_NAMES))
        row.extend(format_metric(value) for value in summary[f'{split_name}_mean'])
        row.extend(format_metric(value) for value in summary[f'{split_name}_std'])
    return row


def write_tsv(path: str, header: List[str], rows: List[List[str]]) -> None:
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as output_f:
        output_f.write('\t'.join(header) + '\n')
        for row in rows:
            output_f.write('\t'.join(row) + '\n')
    os.replace(tmp_path, path)


def read_pid(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as pid_f:
            value = pid_f.read().strip()
        return int(value) if value else None
    except (OSError, ValueError):
        return None


def process_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def apply_state_overrides(summary: Dict[str, object], root_state_dir: str, weight: str, num_runs: int) -> Dict[str, object]:
    safe = safe_weight(weight)
    done_path = os.path.join(root_state_dir, 'done', f'{safe}.done')
    failed_path = os.path.join(root_state_dir, 'failed', f'{safe}.failed')
    lock_dir = os.path.join(root_state_dir, 'locks', f'{safe}.lock')
    lock_pid = read_pid(os.path.join(lock_dir, 'pid'))

    if os.path.exists(done_path) or (len(summary['dev']) >= num_runs and len(summary['test']) >= num_runs):
        summary['status'] = 'done'
    elif process_alive(lock_pid):
        summary['status'] = 'running'
    elif os.path.exists(failed_path):
        summary['status'] = 'failed'
    return summary


def main() -> None:
    args = parse_args()
    tag = dataset_tag(args.dataset, args.mind_size)
    model = model_name(args.news_encoder, args.user_encoder)
    prefix = output_prefix(args.news_encoder, args.user_encoder, args.pairwise_loss)
    root_dir = os.path.join('results', tag, model)
    root_state_dir = state_root(args.news_encoder, args.user_encoder, args.pairwise_loss)
    os.makedirs(root_dir, exist_ok=True)

    header = build_header(args.num_runs)
    rows: List[List[str]] = []
    completed = 0
    for weight in args.weights:
        result_dir = os.path.join(root_dir, f'pairwise_{args.pairwise_loss}_w_{safe_weight(weight)}')
        output_name = f'{prefix}_{weight}.tsv'
        summary = summarize_weight(result_dir, output_name, args.num_runs)
        summary = apply_state_overrides(summary, root_state_dir, weight, args.num_runs)
        write_weight_overview(os.path.join(root_dir, output_name), summary)
        if summary['status'] == 'done':
            completed += 1
        rows.append(build_row(weight, result_dir, summary, args.num_runs))

    summary_path = os.path.join(root_dir, f'{prefix}_sweep_summary.tsv')
    write_tsv(summary_path, header, rows)
    print(f'[summary] completed={completed}/{len(args.weights)} output={summary_path}')


if __name__ == '__main__':
    main()
