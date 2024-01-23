from collections import defaultdict

import numpy as np

raw_file = "/tmp/chandlerzuo/sasrec/ml-1m/ratings.dat"
processed_file = "/tmp/chandlerzuo/sasrec/ml-1m/ml-1m_v3.txt"
max_ih_len = 30
# processed_file = "ml-1m_full.txt"
min_degree = 5
min_user_deg_in_ih = 10


def parse(path):
    g = open(path, "r")
    for l0 in g:
        yield l0.strip().split("::")


def print_distribution(x):
    return (
        f"min {min(x)}"
        f" 10p {np.percentile(x, 10)}"
        f" 25p {np.percentile(x, 25)}"
        f" median {np.median(x)}"
        f" 75p {np.percentile(x, 75)}"
        f" 90p {np.percentile(x, 90)}"
        f" max {max(x)}"
    )


countU = defaultdict(lambda: 0)
line = 0
timestamps = []

for l in parse(raw_file):
    line += 1
    rev, asin, _, timestamp = l
    countU[rev] += 1
    timestamps.append(int(timestamp))

med_timestamp = np.median(timestamps)

countU_train = defaultdict(lambda: 0)
for l in parse(raw_file):
    rev, _, _, timestamp = l
    if int(timestamp) <= med_timestamp:
        countU_train[rev] += 1

countP = defaultdict(lambda: 0)
for l in parse(raw_file):
    line += 1
    rev, asin, _, _ = l
    if countU_train[rev] >= min_user_deg_in_ih:
        countP[asin] += 1

print(f"U_train lengths: {print_distribution(list(countU_train.values()))}")

usermap = {}
usernum = 0
itemmap = {}
itemnum = 0
User = {}
Item = {}
Timestamp = {}
for l in parse(raw_file):
    line += 1
    rev, asin, rating, timestamp = l
    if countU[rev] < min_degree or countP[asin] < min_degree:
        continue

    if rev in usermap:
        userid = usermap[rev]
    else:
        usernum += 1
        userid = usernum
        usermap[rev] = userid
        User[userid] = []
    if asin in itemmap:
        itemid = itemmap[asin]
    else:
        itemnum += 1
        itemid = itemnum
        itemmap[asin] = itemid
        Item[itemid] = []
    User[userid].append([int(timestamp), itemid])
    if countU_train[rev] >= min_user_deg_in_ih:
        Item[itemid].append([int(timestamp), userid])
    Timestamp[f"{userid}_{itemid}"] = int(timestamp)
# sort reviews in User according to time

for userid in User.keys():
    User[userid].sort(key=lambda x: x[0])

for itemid in Item.keys():
    Item[itemid].sort(key=lambda x: x[0])

print(f"{usernum}, {itemnum}")

f = open(processed_file, "w")
nrow = 0
ih_lengths = []
item_cnt_in_train = defaultdict(lambda: 0)
ih_lengths_in_test = defaultdict(lambda: [])
for user in User.keys():
    for timestamp, itemid in User[user]:
        item_history = []
        for x in Item[itemid]:
            if x[0] >= timestamp:
                break
            item_history.append(x[1])
        if len(item_history) > max_ih_len:
            item_history = item_history[-max_ih_len:]
        ih_lengths.append(len(item_history))
        if timestamp <= med_timestamp:
            item_cnt_in_train[itemid] += 1
        else:
            ih_lengths_in_test[itemid].append(len(item_history))
        ih_str = ",".join([str(x) for x in item_history])
        f.write(f"{user} {itemid} {timestamp} {ih_str}\n")
    nrow += 1
    if nrow % 200 == 0:
        print(f"processed {nrow} users")
f.close()

print(f"Item history lengths: {print_distribution(ih_lengths)}")

for group_id in [0, 1]:
    if group_id == 0:
        ih_lens_in_test = {
            k: v for k, v in ih_lengths_in_test.items() if item_cnt_in_train[k] <= 10
        }
    else:
        ih_lens_in_test = {
            k: v for k, v in ih_lengths_in_test.items() if item_cnt_in_train[k] > 10
        }
    all_ih_lens = [x for v in ih_lens_in_test.values() for x in v]
    max_ih_lens = [max(v) for v in ih_lens_in_test.values()]
    print(f"ih lengths in test: {print_distribution(all_ih_lens)}")
    print(f"max ih lengths per item in test: {print_distribution(max_ih_lens)}")
