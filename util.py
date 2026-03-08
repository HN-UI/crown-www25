# -*- coding: utf-8 -*- 
import os
import torch
from torch import Tensor
import torch.nn as nn
from corpus import Corpus
from dataset import DevTest_Dataset
from torch.utils.data import DataLoader
from evaluate import scoring


def compute_scores(model, corpus, batch_size, mode, result_file, dataset, test_filtering_1_1=False, test_filtering_1_0=False, test_filtering_0_0=False, test_filtering_0_1=False):
    assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
    dataloader = DataLoader(DevTest_Dataset(corpus, mode), batch_size=batch_size, shuffle=False, num_workers=batch_size // 16, pin_memory=True)
    indices = (corpus.dev_indices if mode == 'dev' else corpus.test_indices)
    print('[%s] Evaluation sample count : %d' % (mode, len(indices)))
    scores = torch.zeros([len(indices)]).cuda()
    index = 0
    torch.cuda.empty_cache()
    model.eval()
    filtered_test_candidates_1_1 = 0
    filtered_test_candidates_1_0 = 0
    filtered_test_candidates_0_0 = 0
    filtered_test_candidates_0_1 = 0
    with torch.no_grad():
        for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
             news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity) in dataloader:
            user_ID = user_ID.cuda(non_blocking=True)
            user_category = user_category.cuda(non_blocking=True)
            user_subCategory = user_subCategory.cuda(non_blocking=True)
            user_title_text = user_title_text.cuda(non_blocking=True)
            user_title_mask = user_title_mask.cuda(non_blocking=True)
            user_title_entity = user_title_entity.cuda(non_blocking=True)
            user_content_text = user_content_text.cuda(non_blocking=True)
            user_content_mask = user_content_mask.cuda(non_blocking=True)
            user_content_entity = user_content_entity.cuda(non_blocking=True)
            user_history_mask = user_history_mask.cuda(non_blocking=True)
            user_history_graph = user_history_graph.cuda(non_blocking=True)
            user_history_category_mask = user_history_category_mask.cuda(non_blocking=True)
            user_history_category_indices = user_history_category_indices.cuda(non_blocking=True)
            news_category = news_category.cuda(non_blocking=True)
            news_subCategory = news_subCategory.cuda(non_blocking=True)
            news_title_text = news_title_text.cuda(non_blocking=True)
            news_title_mask = news_title_mask.cuda(non_blocking=True)
            news_title_entity = news_title_entity.cuda(non_blocking=True)
            news_content_text = news_content_text.cuda(non_blocking=True)
            news_content_mask = news_content_mask.cuda(non_blocking=True)
            news_content_entity = news_content_entity.cuda(non_blocking=True)
            batch_size = user_ID.size(0)
            news_category = news_category.unsqueeze(dim=1)
            news_subCategory = news_subCategory.unsqueeze(dim=1)
            news_title_text = news_title_text.unsqueeze(dim=1)
            news_title_mask = news_title_mask.unsqueeze(dim=1)
            news_content_text = news_content_text.unsqueeze(dim=1)
            news_content_mask = news_content_mask.unsqueeze(dim=1)
            batch_scores = model(user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_content_text, user_content_mask, user_content_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
                                 news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_content_text, news_content_mask, news_content_entity).squeeze(dim=1) # [batch_size]
            if mode == 'test' and test_filtering_1_1 and hasattr(corpus, 'test_prev_positive_reclicked_flags'):
                flags = torch.tensor(corpus.test_prev_positive_reclicked_flags[index:index+batch_size], device=batch_scores.device, dtype=torch.bool)
                filtered_test_candidates_1_1 += int(flags.sum().item())
                batch_scores = batch_scores.masked_fill(flags, -1e9)
            if mode == 'test' and test_filtering_1_0 and hasattr(corpus, 'test_prev_positive_to_negative_flags'):
                flags = torch.tensor(corpus.test_prev_positive_to_negative_flags[index:index+batch_size], device=batch_scores.device, dtype=torch.bool)
                filtered_test_candidates_1_0 += int(flags.sum().item())
                batch_scores = batch_scores.masked_fill(flags, -1e9)
            if mode == 'test' and test_filtering_0_0 and hasattr(corpus, 'test_prev_negative_to_negative_flags'):
                flags = torch.tensor(corpus.test_prev_negative_to_negative_flags[index:index+batch_size], device=batch_scores.device, dtype=torch.bool)
                filtered_test_candidates_0_0 += int(flags.sum().item())
                batch_scores = batch_scores.masked_fill(flags, -1e9)
            if mode == 'test' and test_filtering_0_1 and hasattr(corpus, 'test_prev_negative_to_positive_flags'):
                flags = torch.tensor(corpus.test_prev_negative_to_positive_flags[index:index+batch_size], device=batch_scores.device, dtype=torch.bool)
                filtered_test_candidates_0_1 += int(flags.sum().item())
                batch_scores = batch_scores.masked_fill(flags, -1e9)
            scores[index: index+batch_size] = batch_scores
            index += batch_size
    if mode == 'test' and test_filtering_1_1:
        print('Filtered test candidates (1->1) : %d' % filtered_test_candidates_1_1)
    if mode == 'test' and test_filtering_1_0:
        print('Filtered test candidates (1->0) : %d' % filtered_test_candidates_1_0)
    if mode == 'test' and test_filtering_0_0:
        print('Filtered test candidates (0->0) : %d' % filtered_test_candidates_0_0)
    if mode == 'test' and test_filtering_0_1:
        print('Filtered test candidates (0->1) : %d' % filtered_test_candidates_0_1)
    scores = scores.tolist()
    sub_scores = [[] for _ in range(indices[-1] + 1)]
    for i, index in enumerate(indices):
        sub_scores[index].append([scores[i], len(sub_scores[index])])
    with open(result_file, 'w', encoding='utf-8') as result_f:
        for i, sub_score in enumerate(sub_scores):
            sub_score.sort(key=lambda x: x[0], reverse=True)
            result = [0 for _ in range(len(sub_score))]
            for j in range(len(sub_score)):
                result[sub_score[j][1]] = j + 1
            result_f.write(('' if i == 0 else '\n') + str(i + 1) + ' ' + str(result).replace(' ', ''))
    if dataset not in ['large', 'mind-large'] or mode != 'test':
        with open(mode + '/ref/truth-%s.txt' % dataset, 'r', encoding='utf-8') as truth_f, open(result_file, 'r', encoding='utf-8') as result_f:
            auc, mrr, ndcg5, ndcg10 = scoring(truth_f, result_f)
        return auc, mrr, ndcg5, ndcg10
    else:
        return None, None, None, None


# result_dir 폴더에서
def get_run_index(result_dir):
    assert os.path.exists(result_dir), 'result directory does not exist' # 폴더 없으면 바로 종료
    max_index = 0
    # 파일 안에서 #로 시작해서 -dev로 끝나는 파일들 찾기 
    for result_file in os.listdir(result_dir):
        if result_file.strip()[0] == '#' and result_file.strip()[-4:] == '-dev':
            index = int(result_file.strip()[1:-4]) # 숫자만 뽑기
            max_index = max(index, max_index) # 가장 큰 숫자만 남기기
    with open(result_dir + '/#' + str(max_index + 1) + '-dev', 'w', encoding='utf-8') as result_f: # 다음 번호에 대한 빈 파일 생성
        pass
    return max_index + 1 # 다음 번호 반환


class AvgMetric:
    def __init__(self, auc, mrr, ndcg5, ndcg10):
        self.auc = auc
        self.mrr = mrr
        self.ndcg5 = ndcg5
        self.ndcg10 = ndcg10
        self.avg = (self.auc + self.mrr + (self.ndcg5 + self.ndcg10) / 2) / 3

    def __gt__(self, value):
        return self.avg > value.avg

    def __ge__(self, value):
        return self.avg >= value.avg

    def __lt__(self, value):
        return self.avg < value.avg

    def __le__(self, value):
        return self.avg <= value.avg

    def __str__(self):
        return '%.4f\nAUC = %.4f\nMRR = %.4f\nnDCG@5 = %.4f\nnDCG@10 = %.4f' % (self.avg, self.auc, self.mrr, self.ndcg5, self.ndcg10)
