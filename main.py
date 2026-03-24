#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import gc
import shutil
from config import Config
import torch
from corpus import Corpus
from model import Model
from aggregate_result import aggregate_model_results, aggregate_overall_test_result, aggregate_result_dir, current_timestamp, split_output_name
from trainer import Trainer, distributed_train
from util import compute_scores, get_run_index
import torch.multiprocessing as mp
from prepare_dataset import preprocess_MIND_small

# function: model training
def train(config, corpus):
    model = Model(config) # NewsRec model object configuration
    model.initialize() # model initialization
    run_index = get_run_index(config.result_dir) 
    # Parallel processing, if distributed training is possible (python 3.7)
    if config.world_size == 1:
        trainer = Trainer(model, config, corpus, run_index)
        trainer.train()
        trainer = None
        del trainer
    else:
        try:
            mp.spawn(distributed_train, args=(model, config, corpus, run_index), nprocs=config.world_size, join=True)
        except Exception as e:
            print(e)
            e = str(e).lower()
            if 'cuda' in e or 'pytorch' in e:
                exit()
    config.run_index = run_index
    model = None
    del model
    gc.collect()
    torch.cuda.empty_cache()

def dev(config, corpus):
    model = Model(config) 
    assert os.path.exists(config.dev_model_path), 'Dev model does not exist : ' + config.dev_model_path
    model.load_state_dict(torch.load(config.dev_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    dev_res_dir = os.path.join(config.dev_res_dir, config.dev_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(dev_res_dir):
        os.mkdir(dev_res_dir)
    auc, mrr, ndcg5, ndcg10 = compute_scores(model, corpus, config.batch_size * 2 // config.world_size, 'dev', dev_res_dir + '/' + model.model_name + '.txt', config.dataset_tag)
    print('Dev : ' + config.dev_model_path)
    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    return auc, mrr, ndcg5, ndcg10

def test(config, corpus):
    model = Model(config) 
    assert os.path.exists(config.test_model_path), 'Test model does not exist : ' + config.test_model_path
    model.load_state_dict(torch.load(config.test_model_path, map_location=torch.device('cpu'))[model.model_name])
    model.cuda()
    test_res_dir = os.path.join(config.test_res_dir, config.test_model_path.replace('\\', '_').replace('/', '_'))
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)
    print('test model path  : ' + config.test_model_path)
    print('test output file : ' + test_res_dir + '/' + model.model_name + '.txt')
    auc, mrr, ndcg5, ndcg10 = compute_scores(
        model, corpus, config.batch_size, 'test', test_res_dir + '/' + model.model_name + '.txt', config.dataset_tag,
        test_filtering_1_1=config.test_filtering_1_1,
        test_filtering_1_0=config.test_filtering_1_0,
        test_filtering_0_0=config.test_filtering_0_0,
        test_filtering_0_1=config.test_filtering_0_1
    )   # config.batch_size * 2

    if auc is None:
        print('Test labels are unavailable for this dataset split. Prediction file was generated only.')
        return

    print('AUC : %.4f\nMRR : %.4f\nnDCG@5 : %.4f\nnDCG@10 : %.4f' % (auc, mrr, ndcg5, ndcg10))
    if config.mode == 'train':
        with open(config.result_dir + '/#' + str(config.run_index) + '-test', 'w') as result_f:
            result_f.write('#' + str(config.run_index) + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')
    elif config.mode == 'test':
        with open(config.test_output_file, 'w', encoding='utf-8') as f:
            f.write('#63' + '\t' + str(auc) + '\t' + str(mrr) + '\t' + str(ndcg5) + '\t' + str(ndcg10) + '\n')


def aggregate_current_model_results(config):
    timestamp = current_timestamp()
    model_name = config.news_encoder + '-' + config.user_encoder
    if getattr(config, 'exp_tag', '') != '':
        labeled_name = model_name + '-' + config.exp_tag
        dev_metrics, test_metrics = aggregate_result_dir(config.result_dir, labeled_name, config.output_name, timestamp)
    else:
        dev_metrics, test_metrics = aggregate_model_results(config.dataset_tag, model_name, config.output_name, timestamp)
        aggregate_overall_test_result(config.dataset_tag, timestamp)

    output_stem, output_ext = split_output_name(config.output_name)
    if dev_metrics is not None:
        print('Aggregated dev results : ' + os.path.join(config.result_dir, output_stem + '-dev' + output_ext))
    if test_metrics is not None:
        print('Aggregated test results : ' + os.path.join(config.result_dir, output_stem + '-test' + output_ext))

# main.py
if __name__ == '__main__':
    config = Config() # configuration
    data_corpus = Corpus(config) # load dataset corpus
    if config.mode == 'train':
        base_seed = config.seed
        for run_offset in range(config.num_runs):
            config.seed = base_seed + run_offset * config.seed_step
            config.attribute_dict['seed'] = config.seed
            config.set_cuda()
            print('[Run %d/%d] seed=%d' % (run_offset + 1, config.num_runs, config.seed))
            train(config, data_corpus)
            config.test_model_path = config.best_model_dir + '/#' + str(config.run_index) + '/' + config.news_encoder + '-' + config.user_encoder
            test(config, data_corpus)
        aggregate_current_model_results(config)
    elif config.mode == 'test':
        config.test_model_path = ''
        config.test_output_file = ''
        test(config, data_corpus)
