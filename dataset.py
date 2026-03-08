from corpus import Corpus
import time
import math
from config import Config
import torch.utils.data as data
import numpy as np
from torch.utils.data import DataLoader
from collections import defaultdict

class Train_Dataset(data.Dataset):
    def __init__(self, corpus: Corpus):
        self.negative_sample_num = corpus.negative_sample_num
        self.drop_repeated_positive_clicks = getattr(corpus, 'drop_repeated_positive_clicks', False)
        self.drop_prev_clicked_from_negatives = getattr(corpus, 'drop_prev_clicked_from_negatives', False)
        self.drop_prev_nonclicked_from_negatives = getattr(corpus, 'drop_prev_nonclicked_from_negatives', False)
        self.repeat_negative_weight = float(getattr(corpus, 'repeat_negative_weight', 1.0))
        self.repeat_negative_sampling_boost = float(getattr(corpus, 'repeat_negative_sampling_boost', 1.0))
        self.repeat_positive_weight = float(getattr(corpus, 'repeat_positive_weight', 1.0))
        self.use_run_length_negative_weight = bool(getattr(corpus, 'use_run_length_negative_weight', False))
        self.use_run_length_positive_weight = bool(getattr(corpus, 'use_run_length_positive_weight', False))
        self.run_length_weight_alpha = float(getattr(corpus, 'run_length_weight_alpha', 0.3))
        self.run_length_weight_beta = float(getattr(corpus, 'run_length_weight_beta', 1.0))
        self.run_length_weight_cap = float(getattr(corpus, 'run_length_weight_cap', 3.0))
        self.positive_run_length_weight_alpha = float(getattr(corpus, 'positive_run_length_weight_alpha', 0.3))
        self.positive_run_length_weight_beta = float(getattr(corpus, 'positive_run_length_weight_beta', 1.0))
        self.positive_run_length_weight_cap = float(getattr(corpus, 'positive_run_length_weight_cap', 5.0))
        # run-length 모드가 켜지면 고정 repeat weight 대신 run-length schedule만 사용
        self.use_repeat_negative_weight = (self.repeat_negative_weight != 1.0) and (not self.use_run_length_negative_weight)
        self.use_repeat_negative_sampling_boost = self.repeat_negative_sampling_boost != 1.0
        self.use_repeat_positive_weight = (self.repeat_positive_weight != 1.0) and (not self.use_run_length_positive_weight)
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.train_user_history_graph
        self.user_history_category_mask = corpus.train_user_history_category_mask
        self.user_history_category_indices = corpus.train_user_history_category_indices

        # 옵션이 켜졌을 때만, 유저별 중복 positive 클릭 샘플을 학습에서 제외
        if self.drop_repeated_positive_clicks:
            self.train_behaviors = self._filter_repeated_positive_behaviors(corpus.train_behaviors)
        else:
            self.train_behaviors = corpus.train_behaviors
        self.train_samples = [[0 for _ in range(1 + self.negative_sample_num)] for __ in range(len(self.train_behaviors))]
        self.train_positive_weights = np.ones([len(self.train_behaviors)], dtype=np.float32)
        self.train_negative_weights = np.ones([len(self.train_behaviors), self.negative_sample_num], dtype=np.float32)
        self.num = len(self.train_behaviors)
        self.prior_clicked_by_behavior = {}
        self.current_clicked_run_length_by_behavior = {}
        self.prior_nonclicked_by_behavior = {}
        self.current_nonclicked_run_length_by_behavior = {}
        if self.drop_prev_clicked_from_negatives or self.use_repeat_positive_weight or self.use_run_length_positive_weight:
            self._build_prior_clicked_map()
        if self.drop_prev_nonclicked_from_negatives or self.use_repeat_negative_weight or self.use_run_length_negative_weight or self.use_repeat_negative_sampling_boost:
            self._build_prior_nonclicked_map()

    def _filter_repeated_positive_behaviors(self, train_behaviors):
        """동일 유저가 과거에 클릭했던 뉴스가 positive로 재등장하면 그 샘플을 제거한다."""
        kept_behaviors = []
        clicked_so_far_by_user = defaultdict(set)
        removed_count = 0
        for train_behavior in train_behaviors:
            user_id = train_behavior[0]
            positive_news = train_behavior[3]
            if positive_news in clicked_so_far_by_user[user_id]:
                removed_count += 1
                continue
            kept_behaviors.append(train_behavior)
            clicked_so_far_by_user[user_id].add(positive_news)
        print('Drop repeated positive clicks: %d -> %d (removed %d)' % (len(train_behaviors), len(kept_behaviors), removed_count))
        return kept_behaviors

    def _build_prior_clicked_map(self):
        """각 impression 시점에서 해당 유저가 과거에 클릭(-1)했던 뉴스 집합을 만든다."""
        behavior_clicked_map = defaultdict(set)
        behavior_order = []
        seen_behaviors = set()
        for train_behavior in self.train_behaviors:
            user_id = train_behavior[0]
            positive_news = train_behavior[3]
            behavior_index = train_behavior[5]
            behavior_key = (user_id, behavior_index)
            if behavior_key not in seen_behaviors:
                seen_behaviors.add(behavior_key)
                behavior_order.append(behavior_key)
            behavior_clicked_map[behavior_key].add(positive_news)

        clicked_so_far_by_user = defaultdict(set)
        current_run_length_by_user = defaultdict(dict) # news_id -> consecutive click run length
        for behavior_key in behavior_order:
            user_id, _ = behavior_key
            current_clicked = behavior_clicked_map[behavior_key]
            self.prior_clicked_by_behavior[behavior_key] = set(clicked_so_far_by_user[user_id])
            clicked_so_far_by_user[user_id].update(current_clicked)

            prev_run_lengths = current_run_length_by_user[user_id]
            run_lengths_this_behavior = {}
            for news_id in current_clicked:
                run_lengths_this_behavior[news_id] = prev_run_lengths.get(news_id, 0) + 1
            self.current_clicked_run_length_by_behavior[behavior_key] = run_lengths_this_behavior
            current_run_length_by_user[user_id] = run_lengths_this_behavior

    def _build_prior_nonclicked_map(self):
        """각 impression 시점에서 해당 유저가 과거에 비클릭(-0)으로 노출된 뉴스 집합을 만든다."""
        behavior_nonclicked_map = defaultdict(set)
        behavior_order = []
        seen_behaviors = set()
        for train_behavior in self.train_behaviors:
            user_id = train_behavior[0]
            behavior_index = train_behavior[5]
            behavior_key = (user_id, behavior_index)
            if behavior_key not in seen_behaviors:
                seen_behaviors.add(behavior_key)
                behavior_order.append(behavior_key)
            behavior_nonclicked_map[behavior_key].update(train_behavior[4])

        nonclicked_so_far_by_user = defaultdict(set)
        current_run_length_by_user = defaultdict(dict) # news_id -> consecutive non-click run length
        for behavior_key in behavior_order:
            user_id, _ = behavior_key
            current_nonclicked = behavior_nonclicked_map[behavior_key]
            self.prior_nonclicked_by_behavior[behavior_key] = set(nonclicked_so_far_by_user[user_id])
            nonclicked_so_far_by_user[user_id].update(current_nonclicked)

            prev_run_lengths = current_run_length_by_user[user_id]
            run_lengths_this_behavior = {}
            for news_id in current_nonclicked:
                run_lengths_this_behavior[news_id] = prev_run_lengths.get(news_id, 0) + 1
            self.current_nonclicked_run_length_by_behavior[behavior_key] = run_lengths_this_behavior
            current_run_length_by_user[user_id] = run_lengths_this_behavior

    def _compute_run_length_weight(self, run_length: int) -> float:
        if run_length <= 1:
            return 1.0
        weight = 1.0 + self.run_length_weight_beta + self.run_length_weight_alpha * math.log2(run_length - 1)
        weight = max(1.0, weight)
        return min(self.run_length_weight_cap, weight)

    def _compute_positive_run_length_weight(self, run_length: int) -> float:
        if run_length <= 1:
            return 1.0
        weight = 1.0 + self.positive_run_length_weight_beta + self.positive_run_length_weight_alpha * math.log2(run_length - 1)
        weight = max(1.0, weight)
        return min(self.positive_run_length_weight_cap, weight)

    def _build_negative_sampling_probabilities(self, negative_samples, prior_nonclicked):
        if not self.use_repeat_negative_sampling_boost or not prior_nonclicked:
            return None, 0
        weights = np.ones([len(negative_samples)], dtype=np.float64)
        boosted_candidate_count = 0
        for idx, news_id in enumerate(negative_samples):
            if news_id in prior_nonclicked:
                weights[idx] = self.repeat_negative_sampling_boost
                boosted_candidate_count += 1
        if boosted_candidate_count == 0:
            return None, 0
        weight_sum = weights.sum()
        if (not np.isfinite(weight_sum)) or weight_sum <= 0:
            return None, 0
        return weights / weight_sum, boosted_candidate_count

    def _sample_negative_indices(self, news_num, probabilities):
        if news_num <= self.negative_sample_num:
            if probabilities is None:
                return [j % news_num for j in range(self.negative_sample_num)]
            return np.random.choice(news_num, size=self.negative_sample_num, replace=True, p=probabilities).tolist()
        if probabilities is None:
            return np.random.choice(news_num, size=self.negative_sample_num, replace=False).tolist()
        return np.random.choice(news_num, size=self.negative_sample_num, replace=False, p=probabilities).tolist()

    def _assign_negative_sample(self, behavior_idx, negative_slot_idx, sampled_negative, prior_nonclicked, current_nonclicked_run_lengths):
        self.train_samples[behavior_idx][negative_slot_idx + 1] = sampled_negative
        assigned_weight = 1.0
        if self.use_run_length_negative_weight:
            run_length = current_nonclicked_run_lengths.get(sampled_negative, 1)
            if run_length >= 2:
                assigned_weight = self._compute_run_length_weight(run_length)
                self.train_negative_weights[behavior_idx][negative_slot_idx] = assigned_weight
                return True, assigned_weight
        elif self.use_repeat_negative_weight and sampled_negative in prior_nonclicked:
            assigned_weight = self.repeat_negative_weight
            self.train_negative_weights[behavior_idx][negative_slot_idx] = assigned_weight
            return True, assigned_weight
        return False, assigned_weight

    def negative_sampling(self, rank=None):
        print('\n%sBegin negative sampling, training sample num : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), self.num))
        start_time = time.time()
        filtered_by_prior_clicked = 0
        filtered_by_prior_nonclicked = 0
        weighted_repeated_positives = 0
        max_assigned_positive_weight = 1.0
        weighted_repeated_negatives = 0
        max_assigned_negative_weight = 1.0
        sampling_boosted_negatives = 0
        sampling_boost_affected_impressions = 0
        for i, train_behavior in enumerate(self.train_behaviors):
            positive_news = train_behavior[3]
            self.train_samples[i][0] = positive_news
            self.train_positive_weights[i] = 1.0
            self.train_negative_weights[i].fill(1.0)
            negative_samples = train_behavior[4]
            behavior_key = (train_behavior[0], train_behavior[5])
            prior_clicked = self.prior_clicked_by_behavior.get(behavior_key, set())
            current_clicked_run_lengths = self.current_clicked_run_length_by_behavior.get(behavior_key, {})
            prior_nonclicked = self.prior_nonclicked_by_behavior.get(behavior_key, set())
            current_nonclicked_run_lengths = self.current_nonclicked_run_length_by_behavior.get(behavior_key, {})

            if self.use_run_length_positive_weight:
                run_length = current_clicked_run_lengths.get(positive_news, 1)
                if run_length >= 2:
                    assigned_weight = self._compute_positive_run_length_weight(run_length)
                    self.train_positive_weights[i] = assigned_weight
                    weighted_repeated_positives += 1
                    max_assigned_positive_weight = max(max_assigned_positive_weight, assigned_weight)
            elif self.use_repeat_positive_weight and positive_news in prior_clicked:
                self.train_positive_weights[i] = self.repeat_positive_weight
                weighted_repeated_positives += 1
                max_assigned_positive_weight = max(max_assigned_positive_weight, self.repeat_positive_weight)

            if self.drop_prev_clicked_from_negatives:
                if prior_clicked:
                    original_len = len(negative_samples)
                    negative_samples = [n for n in negative_samples if n not in prior_clicked]
                    if len(negative_samples) != original_len:
                        filtered_by_prior_clicked += 1
            if self.drop_prev_nonclicked_from_negatives:
                if prior_nonclicked:
                    original_len = len(negative_samples)
                    negative_samples = [n for n in negative_samples if n not in prior_nonclicked]
                    if len(negative_samples) != original_len:
                        filtered_by_prior_nonclicked += 1
            news_num = len(negative_samples)
            if news_num == 0:
                # 후보가 비면 PAD(0)으로 채워 텐서 shape 유지
                for j in range(self.negative_sample_num):
                    self.train_samples[i][j + 1] = 0
                continue
            sampling_probabilities, boosted_candidate_count = self._build_negative_sampling_probabilities(negative_samples, prior_nonclicked)
            if boosted_candidate_count > 0:
                sampling_boost_affected_impressions += 1
            sampled_indices = self._sample_negative_indices(news_num, sampling_probabilities)
            for j, sampled_idx in enumerate(sampled_indices):
                sampled_negative = negative_samples[sampled_idx]
                if boosted_candidate_count > 0 and sampled_negative in prior_nonclicked:
                    sampling_boosted_negatives += 1
                is_weighted_negative, assigned_weight = self._assign_negative_sample(
                    i,
                    j,
                    sampled_negative,
                    prior_nonclicked,
                    current_nonclicked_run_lengths,
                )
                if is_weighted_negative:
                    weighted_repeated_negatives += 1
                    max_assigned_negative_weight = max(max_assigned_negative_weight, assigned_weight)
        end_time = time.time()
        if self.drop_prev_clicked_from_negatives:
            print('%sFiltered negatives by prior clicks, affected impressions : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), filtered_by_prior_clicked))
        if self.drop_prev_nonclicked_from_negatives:
            print('%sFiltered negatives by prior non-clicks, affected impressions : %d' % ('' if rank is None else ('rank ' + str(rank) + ' : '), filtered_by_prior_nonclicked))
        if self.use_repeat_negative_sampling_boost:
            print(
                '%sSampling-boosted repeated 0->0 negatives : %d (boost=%.3f, affected_impressions=%d)' %
                (
                    '' if rank is None else ('rank ' + str(rank) + ' : '),
                    sampling_boosted_negatives,
                    self.repeat_negative_sampling_boost,
                    sampling_boost_affected_impressions,
                )
            )
        if self.use_run_length_positive_weight:
            print(
                '%sWeighted repeated 1->1 positives : %d (run-length schedule, alpha=%.3f, beta=%.3f, cap=%.3f, max_assigned=%.3f)' %
                (
                    '' if rank is None else ('rank ' + str(rank) + ' : '),
                    weighted_repeated_positives,
                    self.positive_run_length_weight_alpha,
                    self.positive_run_length_weight_beta,
                    self.positive_run_length_weight_cap,
                    max_assigned_positive_weight,
                )
            )
        elif self.use_repeat_positive_weight:
            print('%sWeighted repeated 1->1 positives : %d (weight=%.3f)' % ('' if rank is None else ('rank ' + str(rank) + ' : '), weighted_repeated_positives, self.repeat_positive_weight))
        if self.use_run_length_negative_weight:
            print(
                '%sWeighted repeated 0->0 negatives : %d (run-length schedule, alpha=%.3f, beta=%.3f, cap=%.3f, max_assigned=%.3f)' %
                (
                    '' if rank is None else ('rank ' + str(rank) + ' : '),
                    weighted_repeated_negatives,
                    self.run_length_weight_alpha,
                    self.run_length_weight_beta,
                    self.run_length_weight_cap,
                    max_assigned_negative_weight,
                )
            )
        elif self.use_repeat_negative_weight:
            print('%sWeighted repeated 0->0 negatives : %d (weight=%.3f)' % ('' if rank is None else ('rank ' + str(rank) + ' : '), weighted_repeated_negatives, self.repeat_negative_weight))
        print('%sEnd negative sampling, used time : %.3fs' % ('' if rank is None else ('rank ' + str(rank) + ' : '), end_time - start_time))

    # user_ID                       : [1]
    # user_category                 : [max_history_num]
    # usre_subCategory              : [max_history_num]
    # user_title_text               : [max_history_num, max_title_length]
    # user_title_mask               : [max_history_num, max_title_length]
    # user_title_entity             : [max_history_num, max_title_length]
    # user_abstract_text            : [max_history_num, max_abstract_length]
    # user_abstract_mask            : [max_history_num, max_abstract_length]
    # user_abstract_entity          : [max_history_num, max_abstract_length]
    # user_history_mask             : [max_history_num]
    # user_history_graph            : [max_history_num, max_history_num]
    # user_history_category_mask    : [category_num + 1]
    # user_history_category_indices : [max_history_num]
    # news_category                 : [1 + negative_sample_num]
    # news_subCategory              : [1 + negative_sample_num]
    # news_title_text               : [1 + negative_sample_num, max_title_length]
    # news_title_mask               : [1 + negative_sample_num, max_title_length]
    # news_title_entity             : [1 + negative_sample_num, max_title_length]
    # news_abstract_text            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_mask            : [1 + negative_sample_num, max_abstract_length]
    # news_abstract_entity          : [1 + negative_sample_num, max_abstract_length]
    # positive_weight               : [1]
    # negative_weights              : [negative_sample_num]

    def __getitem__(self, index):
        train_behavior = self.train_behaviors[index]
        history_index = train_behavior[1]
        sample_index = self.train_samples[index]
        behavior_index = train_behavior[5]
        return train_behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], train_behavior[2], self.user_history_graph[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
               self.news_category[sample_index], self.news_subCategory[sample_index], self.news_title_text[sample_index], self.news_title_mask[sample_index], self.news_title_entity[sample_index], self.news_abstract_text[sample_index], self.news_abstract_mask[sample_index], self.news_abstract_entity[sample_index], self.train_positive_weights[index], self.train_negative_weights[index]

    def __len__(self):
        return self.num


class DevTest_Dataset(data.Dataset):
    def __init__(self, corpus: Corpus, mode: str):
        assert mode in ['dev', 'test'], 'mode must be chosen from \'dev\' or \'test\''
        self.news_category = corpus.news_category
        self.news_subCategory = corpus.news_subCategory
        self.news_title_text =  corpus.news_title_text
        self.news_title_mask = corpus.news_title_mask
        self.news_title_entity = corpus.news_title_entity
        self.news_abstract_text =  corpus.news_abstract_text
        self.news_abstract_mask = corpus.news_abstract_mask
        self.news_abstract_entity = corpus.news_abstract_entity
        self.user_history_graph = corpus.dev_user_history_graph if mode == 'dev' else corpus.test_user_history_graph
        self.user_history_category_mask = corpus.dev_user_history_category_mask if mode == 'dev' else corpus.test_user_history_category_mask
        self.user_history_category_indices = corpus.dev_user_history_category_indices if mode == 'dev' else corpus.test_user_history_category_indices
        self.behaviors = corpus.dev_behaviors if mode == 'dev' else corpus.test_behaviors
        self.num = len(self.behaviors)

    # user_ID                        : [1]
    # user_category                  : [max_history_num]
    # user_subCategory               : [max_history_num]
    # user_title_text                : [max_history_num, max_title_length]
    # user_title_mask                : [max_history_num, max_title_length]
    # user_title_entity              : [max_history_num, max_title_length]
    # user_abstract_text             : [max_history_num, max_abstract_length]
    # user_abstract_mask             : [max_history_num, max_abstract_length]
    # user_abstract_entity           : [max_history_num, max_abstract_length]
    # user_history_mask              : [max_history_num]
    # user_history_graph             : [max_history_num, max_history_num]
    # user_history_category_mask     : [category_num + 1]
    # user_history_category_indices  : [max_history_num]
    # candidate_news_category        : [1]
    # candidate_news_subCategory     : [1]
    # candidate_news_title_text      : [max_title_length]
    # candidate_news_title_mask      : [max_title_length]
    # candidate_news_title_entity    : [max_title_lenght]
    # candidate_news_abstract_text   : [max_abstract_length]
    # candidate_news_abstract_mask   : [max_abstract_length]
    # candidate_news_abstract_entity : [max_abstract_length]
    def __getitem__(self, index):
        behavior = self.behaviors[index]
        history_index = behavior[1]
        candidate_news_index = behavior[3]
        behavior_index = behavior[4]
        return behavior[0], self.news_category[history_index], self.news_subCategory[history_index], self.news_title_text[history_index], self.news_title_mask[history_index], self.news_title_entity[history_index], self.news_abstract_text[history_index], self.news_abstract_mask[history_index], self.news_abstract_entity[history_index], behavior[2], self.user_history_graph[behavior_index], self.user_history_category_mask[behavior_index], self.user_history_category_indices[behavior_index], \
               self.news_category[candidate_news_index], self.news_subCategory[candidate_news_index], self.news_title_text[candidate_news_index], self.news_title_mask[candidate_news_index], self.news_title_entity[candidate_news_index], self.news_abstract_text[candidate_news_index], self.news_abstract_mask[candidate_news_index], self.news_abstract_entity[candidate_news_index]

    def __len__(self):
        return self.num

if __name__ == '__main__':
    start_time = time.time()
    config = Config()
    dataset_corpus = Corpus(config)
    print('user_num :', len(dataset_corpus.user_ID_dict))
    print('news_num :', len(dataset_corpus.news_title_text))
    print('average title word num :', dataset_corpus.title_word_num / dataset_corpus.news_num)
    print('average abstract word num :', dataset_corpus.abstract_word_num / dataset_corpus.news_num)
    train_dataset = Train_Dataset(dataset_corpus)
    dev_dataset = DevTest_Dataset(dataset_corpus, 'dev')
    test_dataset = DevTest_Dataset(dataset_corpus, 'test')
    train_dataset.negative_sampling()
    end_time = time.time()
    print('load time : %.3fs' % (end_time - start_time))
    print('Train_Dataset :', len(train_dataset))
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity, positive_weights, negative_weights) in train_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        print('positive_weights', positive_weights.size(), positive_weights.dtype)
        print('negative_weights', negative_weights.size(), negative_weights.dtype)
        break
    print('Dev_Dataset :', len(dev_dataset))
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity) in dev_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        break
    print(len(dataset_corpus.dev_indices))
    print('Test_Dataset :', len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.batch_size // 16)
    for (user_ID, user_category, user_subCategory, user_title_text, user_title_mask, user_title_entity, user_abstract_text, user_abstract_mask, user_abstract_entity, user_history_mask, user_history_graph, user_history_category_mask, user_history_category_indices, \
         news_category, news_subCategory, news_title_text, news_title_mask, news_title_entity, news_abstract_text, news_abstract_mask, news_abstract_entity) in test_dataloader:
        print('user_ID', user_ID.size(), user_ID.dtype)
        print('user_category', user_category.size(), user_category.dtype)
        print('user_subCategory', user_subCategory.size(), user_subCategory.dtype)
        print('user_title_text', user_title_text.size(), user_title_text.dtype)
        print('user_title_mask', user_title_mask.size(), user_title_mask.dtype)
        print('user_title_entity', user_title_entity.size(), user_title_entity.dtype)
        print('user_abstract_text', user_abstract_text.size(), user_abstract_text.dtype)
        print('user_abstract_mask', user_abstract_mask.size(), user_abstract_mask.dtype)
        print('user_abstract_entity', user_abstract_entity.size(), user_abstract_entity.dtype)
        print('user_history_mask', user_history_mask.size(), user_history_mask.dtype)
        print('user_history_graph', user_history_graph.size(), user_history_graph.dtype)
        print('user_history_category_mask', user_history_category_mask.size(), user_history_category_mask.dtype)
        print('user_history_category_indices', user_history_category_indices.size(), user_history_category_indices.dtype)
        print('news_category', news_category.size(), news_category.dtype)
        print('news_subCategory', news_subCategory.size(), news_subCategory.dtype)
        print('news_title_text', news_title_text.size(), news_title_text.dtype)
        print('news_title_mask', news_title_mask.size(), news_title_mask.dtype)
        print('news_title_entity', news_title_entity.size(), news_title_entity.dtype)
        print('news_abstract_text', news_abstract_text.size(), news_abstract_text.dtype)
        print('news_abstract_mask', news_abstract_mask.size(), news_abstract_mask.dtype)
        print('news_abstract_entity', news_abstract_entity.size(), news_abstract_entity.dtype)
        break
    print(len(dataset_corpus.test_indices))
