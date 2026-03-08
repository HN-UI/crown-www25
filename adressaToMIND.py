# Adressa 데이터 MIND 형식으로 바꾸기
import json
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--adressa_news_path",
        type=str,
        default="/home/yyko/workspace/news_rec/preprocess_Adressa/AdressaDataLSTMRecommender-master/one_week/home/lemeiz/content_refine",
        help="path to downloaded adressa news dataset(title + body information)",
    )
    parser.add_argument(
        "--adressa_user_path",
        type=str,
        default="/home/yyko/workspace/news_rec/[IJCAI2020]HyperNews-main/data/one_week",
        help="path to downloaded raw adressa dataset",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="/home/yyko/workspace/news_rec/Adressa-dwellTime",
        help="path to save processed dataset, default in ../Adressa",
    )
    parser.add_argument(
        "--neg_num",
        type=int,
        default=20,
        help="randomly sample neg_num negative impression for every positive behavior",
    )

    args = parser.parse_args()
    return args

def process_news(adressa_news_path):
    news_title = {}
    news_body = {}    
    news_category = {}
    news_subCategory = {}

    for file in adressa_news_path.iterdir():
        with open(file, "r") as f:
            for l in f:
                data = json.loads(l)
                id = data["id"]
                fields_list = data["fields"]
                title_dict = next((item for item in fields_list if item["field"] == "title"), None)
                category_dict = next((item for item in fields_list if item["field"] == "kw-category"), None)                
                body_dict = next((item for item in fields_list if item["field"] == "body"), None)
                if title_dict and category_dict and body_dict:
                    news_title[id] = title_dict["value"]
                    first5sentences = body_dict["value"][:5]
                    first5sentences_str = ' '.join(first5sentences)
                    body_0 = first5sentences_str.replace('\n', ' ')
                    body_str = body_0.replace('\t', ' ')
                    news_body[id] = body_str
                    if isinstance(category_dict["value"], str):
                        news_category[id] = category_dict["value"]
                        news_subCategory[id] = ""
                    else:
                        news_category[id] = category_dict["value"][0]
                        news_subCategory[id] = category_dict["value"][1]
                    break
                else: 
                    break
    nid2index = {k: v for k, v in zip(news_title.keys(), range(1, len(news_title) + 1))}
    return news_title, news_body, news_category, news_subCategory, nid2index

def write_news_files(news_title, news_body, news_category, news_subCategory, nid2index, out_path):
    # Output with MIND format
    news_lines = []
    for nid in tqdm(news_title):
        nindex = nid2index[nid]
        title = news_title[nid]
        body = news_body[nid]
        category = news_category[nid]
        subCategory = news_subCategory[nid]
        
        news_line = "\t".join([str(nindex), category, subCategory, title, body, "", "", ""]) + "\n"
        news_lines.append(news_line)

    for stage in ["train", "dev", "test"]:
        file_path = out_path / stage
        file_path.mkdir(exist_ok=True, parents=True)
        with open(out_path / stage / "news.tsv", "w", encoding="utf-8") as f:
            f.writelines(news_lines)


class UserInfo:
    def __init__(self, train_day=6, test_day=7):
        self.click_news = []
        self.click_time = []
        self.click_activeTime = []
        self.click_days = []

        self.train_news = []
        self.train_time = []
        self.train_activeTime = []
        self.train_days = []

        self.test_news = []
        self.test_time = []
        self.test_activeTime = []
        self.test_days = []

        self.train_day = train_day
        self.test_day = test_day

    def update(self, nindex, time, activeTime, day):
        if day == self.train_day:
            self.train_news.append(nindex)
            self.train_time.append(time)
            self.train_activeTime.append(activeTime)
            self.train_days.append(day)
        elif day == self.test_day:
            self.test_news.append(nindex)
            self.test_time.append(time)
            self.test_activeTime.append(activeTime)
            self.test_days.append(day)
        else:
            self.click_news.append(nindex)
            self.click_time.append(time)
            self.click_activeTime.append(activeTime)
            self.click_days.append(day)

    def sort_click(self):
        self.click_news = np.array(self.click_news, dtype="int32")
        self.click_time = np.array(self.click_time, dtype="int32")
        self.click_activeTime = np.array(self.click_activeTime, dtype="int32")
        self.click_days = np.array(self.click_days, dtype="int32")

        self.train_news = np.array(self.train_news, dtype="int32")
        self.train_time = np.array(self.train_time, dtype="int32")
        self.train_activeTime = np.array(self.train_activeTime, dtype="int32")
        self.train_days = np.array(self.train_days, dtype="int32")

        self.test_news = np.array(self.test_news, dtype="int32")
        self.test_time = np.array(self.test_time, dtype="int32")
        self.test_activeTime = np.array(self.test_activeTime, dtype="int32")
        self.test_days = np.array(self.test_days, dtype="int32")

        order = np.argsort(self.train_time)
        self.train_time = self.train_time[order]
        self.train_days = self.train_days[order]
        self.train_news = self.train_news[order]
        self.train_activeTime = self.train_activeTime[order]

        order = np.argsort(self.test_time)
        self.test_time = self.test_time[order]
        self.test_days = self.test_days[order]
        self.test_news = self.test_news[order]
        self.test_activeTime = self.test_activeTime[order]

        order = np.argsort(self.click_time)
        self.click_time = self.click_time[order]
        self.click_days = self.click_days[order]
        self.click_news = self.click_news[order]
        self.click_activeTime = self.click_activeTime[order]


import random
random.seed(2023)
def process_users(adressa_user_path, news_title, nid2index):
    uid2index = {}
    user_info = defaultdict(UserInfo)

    for file in adressa_user_path.iterdir():
        with open(file, "r") as f:
            for l in tqdm(f):
                event_dict = json.loads(l.strip("\n"))
                if "title" in event_dict:            
                    matching_id = [key for key, value in news_title.items() if value == event_dict["title"]]
                    if len(matching_id) != 0: 
                        nindex = nid2index[matching_id[0]]
                        uid = event_dict["userId"]
                        
                        if "activeTime" in event_dict:
                            activeTime = int(event_dict["activeTime"])
                        else:
                            activeTime = 0
                        
                        if uid not in uid2index:
                            uid2index[uid] = len(uid2index)
                        uindex = uid2index[uid]
                        click_time = int(event_dict["time"])
                        day = int(file.name[-1])
                        user_info[uindex].update(nindex, click_time, activeTime, day)

    return uid2index, user_info


def construct_behaviors(uindex, click_news, train_news, test_news, click_activeTime, train_activeTime, test_activeTime, neg_num):
    p = np.ones(len(news_title) + 1, dtype="float32")
    p[click_news] = 0
    p[train_news] = 0
    p[test_news] = 0
    p[0] = 0
    p /= p.sum()
    
    train_his_news = [str(i) for i in click_news.tolist()]
    train_his_line = " ".join(train_his_news)
    train_his_news_activeTime = [int(i) for i in click_activeTime.tolist()]

    for pos_nindex, pos_activeTime in zip(train_news, train_activeTime):
        neg_cand = np.random.choice(
            len(news_title) + 1, size=neg_num, replace=False, p=p
        ).tolist()
        cand_news = " ".join(
            [f"{str(pos_nindex)}-1"] + [f"{str(neg_nindex)}-0" for neg_nindex in neg_cand]
        )
        activeTime_list = [train_his_news_activeTime, [pos_activeTime]]
        train_behavior_line = f"null\t{uindex}\t{activeTime_list}\t{train_his_line}\t{cand_news}\n"
        train_lines.append(train_behavior_line)

    test_his_news = [str(i) for i in click_news.tolist() + train_news.tolist()]
    test_his_line = " ".join(test_his_news)
    test_his_news_activeTime = [int(i) for i in click_activeTime.tolist() + train_activeTime.tolist()]
    
    for pos_nindex, pos_activeTime in zip(test_news, test_activeTime):
        neg_cand = np.random.choice(
            len(news_title) + 1, size=neg_num, replace=False, p=p
        ).tolist()
        cand_news = " ".join(
            [f"{str(pos_nindex)}-1"] + [f"{str(neg_nindex)}-0" for neg_nindex in neg_cand]
        )
        activeTime_list = [test_his_news_activeTime, [pos_activeTime]]
        test_behavior_line = f"null\t{uindex}\t{activeTime_list}\t{test_his_line}\t{cand_news}\n"
        test_lines.append(test_behavior_line)


if __name__ == "__main__":
    args = parse_args()
    adressa_news_path = Path(args.adressa_news_path)
    adressa_user_path = Path(args.adressa_user_path)
    out_path = Path(args.out_path)

    news_title, news_body, news_category, news_subCategory, nid2index = process_news(adressa_news_path)
    write_news_files(news_title, news_body, news_category, news_subCategory, nid2index, out_path)

    uid2index, user_info = process_users(adressa_user_path, news_title, nid2index)
    
    for uid in tqdm(user_info):
        user_info[uid].sort_click()

    train_lines = []
    test_lines = []
    for uindex in tqdm(user_info):
        uinfo = user_info[uindex]
        click_news = uinfo.click_news
        train_news = uinfo.train_news
        test_news = uinfo.test_news
        click_activeTime = uinfo.click_activeTime
        train_activeTime = uinfo.train_activeTime
        test_activeTime = uinfo.test_activeTime
        
        construct_behaviors(uindex, click_news, train_news, test_news, click_activeTime, train_activeTime, test_activeTime, args.neg_num)

    test_split_lines, valid_split_lines = train_test_split(
        test_lines, test_size=0.3, random_state=77
    )
    with open(out_path / "train" / "behaviors.tsv", "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(out_path / "dev" / "behaviors.tsv", "w", encoding="utf-8") as f:
        f.writelines(valid_split_lines)

    with open(out_path / "test" / "behaviors.tsv", "w", encoding="utf-8") as f:
        f.writelines(test_split_lines)
