# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from collections import defaultdict

import numpy as np

# buck run @//mode/opt :data_distribution


def check_distribution(
    processed_file: str, num_test_groups: int = 5, longtail_cutoff: int = 10
):
    print(processed_file)
    User = defaultdict(list)
    usernum = 0
    itemnum = 0
    all_ts = []

    with open(processed_file, "r") as f:
        for line in f:
            u, i, ts, ih = line.strip("\n").split(" ")
            u = int(u)
            i = int(i)
            ts = int(ts)
            ih = [int(x) for x in ih.split(",")] if len(ih) > 0 else []
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append((i, ih, ts))
            all_ts.append(ts)
    med_ts = np.median(all_ts)
    max_ts = np.max(all_ts)
    step_size = (max_ts - med_ts) / num_test_groups
    cutoffs = [med_ts + i * step_size for i in range(num_test_groups)]
    cutoffs.append(max_ts)
    print(f"cutoff ts values: {cutoffs}")

    item_cnt_train = {}
    for user in User:
        for i, _, ts in User[user]:
            if ts <= cutoffs[0]:
                item_cnt_train.setdefault(i, 0)
                item_cnt_train[i] += 1

    head_items = [
        k for k in item_cnt_train.keys() if item_cnt_train[k] >= longtail_cutoff
    ]
    tail_items = [
        k for k in item_cnt_train.keys() if item_cnt_train[k] < longtail_cutoff
    ]
    print(f"head items: {len(head_items)} tail items: {len(tail_items)}")

    for i_test in range(num_test_groups):
        print(f"Test group {i_test}")
        item_cnt_test = {}
        item_ih_len_total = {}
        for user in User:
            for i, ih, ts in User[user]:
                if ts <= cutoffs[i_test + 1] and ts > cutoffs[i_test]:
                    item_cnt_test.setdefault(i, 0)
                    item_ih_len_total.setdefault(i, 0)
                    item_cnt_test[i] += 1
                    item_ih_len_total[i] += len(ih)
        head_items = [
            k
            for k in item_cnt_test.keys()
            if item_cnt_train.get(k, 0) >= longtail_cutoff
        ]
        avg_head_item_ih_len = (
            sum(item_ih_len_total[item_id] for item_id in head_items)
            * 1.0
            / sum(item_cnt_test[item_id] for item_id in head_items)
        )
        tail_items = [
            k
            for k in item_cnt_test.keys()
            if item_cnt_train.get(k, 0) < longtail_cutoff
        ]
        avg_tail_item_ih_len = (
            sum(item_ih_len_total[item_id] for item_id in tail_items)
            * 1.0
            / sum(item_cnt_test[item_id] for item_id in tail_items)
        )
        print(
            f"head items: {len(head_items)} avg ih length {avg_head_item_ih_len}"
            f" tail items: {len(tail_items)} avg ih length {avg_tail_item_ih_len}"
        )


if __name__ == "__main__":
    for fname in [
        "ml-1m/ml-1m.txt",
        "amazon/amazon_books.txt",
        "amazon/amazon_games.txt",
        "amazon/amazon_beauty.txt",
    ]:
        check_distribution(fname)
