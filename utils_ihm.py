import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, maxlen_ih, result_queue, SEED):
    # construct negative sampling pool
    # the last element in each user sequence is a (itemid, item_history) tuple
    negpool = [x[-1] for x in user_train.values()]
    negpool_i_to_ih = {}
    for i, ih in negpool:
        if i not in negpool_i_to_ih: negpool_i_to_ih[i] = []
        negpool_i_to_ih[i].append(ih)
    print(f"{len(negpool_i_to_ih)} items in the neg pool")

    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train.get(user, [])) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        pos_ih = np.zeros([maxlen, maxlen_ih], dtype=np.int32)
        neg_ih = np.zeros([maxlen, maxlen_ih], dtype=np.int32)
        nxt, nxt_ih = user_train[user][-1]
        idx = maxlen - 1

        true_set = set([itemid for itemid, _ in user_train[user]])
        for i, ih in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            nxt_ih_len = min(maxlen_ih, len(nxt_ih))
            nxt_ih = [0] * (maxlen_ih - nxt_ih_len) + nxt_ih[-nxt_ih_len:]
            pos_ih[idx, :] = nxt_ih
            if nxt != 0:
                """
                # pool negative sampling
                neg_pool_idx = np.random.randint(0, len(negpool))
                neg_spl_i, neg_spl_ih = negpool[neg_pool_idx]
                while neg_spl_i in true_set:
                    neg_spl_idx = np.random.randint(0, len(negpool))
                    neg_spl_i, neg_spl_ih = negpool[neg_spl_idx]
                # end of pool negative sampling
                """
                # uniform negative sampling
                neg_spl_i = np.random.randint(1, itemnum + 1)
                while neg_spl_i in true_set:
                    neg_spl_i = np.random.randint(1, itemnum + 1)
                # random sample a history sequence for this item
                if neg_spl_i in negpool_i_to_ih:
                    neg_spl_ih_idx = np.random.randint(0, len(negpool_i_to_ih[neg_spl_i]))
                    neg_spl_ih = negpool_i_to_ih[neg_spl_i][neg_spl_ih_idx]
                else:
                    neg_spl_ih = []
                # end of uniform negative sampling
                neg_spl_ih_len = min(maxlen_ih, len(neg_spl_ih))
                neg_spl_ih = [0] * (maxlen_ih - neg_spl_ih_len) + neg_spl_ih[-neg_spl_ih_len:]
                neg[idx], neg_ih[idx, :] = neg_spl_i, neg_spl_ih
            nxt = i
            nxt_ih = ih
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg, pos_ih, neg_ih)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, maxlen_ih=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      maxlen_ih,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('ml-1m/%s.txt' % fname, 'r')
    for line in f:
        u, i, _, ih = line.strip('\n').split(' ')
        u = int(u)
        i = int(i)
        ih = [int(x) for x in ih.split(',')] if len(ih) > 0 else []
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append((i, ih))

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

def data_partition_by_ts(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    # assume user/item index starting from 1
    all_ts = []
    with open('ml-1m/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i, ts, ih = line.strip('\n').split(' ')
            u = int(u)
            i = int(i)
            ts = int(ts)
            ih = [int(x) for x in ih.split(',')] if len(ih) > 0 else []
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append((i, ih, ts))
            all_ts.append(ts)
    med_ts = np.median(all_ts)
    max_ts = np.max(all_ts)
    num_test_groups = 5
    step_size = (max_ts - med_ts) / num_test_groups
    cutoffs = [med_ts + i * step_size for i in range(num_test_groups)]
    cutoffs.append(max_ts)
    print(f"cutoff ts values: {cutoffs}")

    user_train = {}
    user_test = [{} for _ in range(num_test_groups)]
    for user in User:
        nfeedback = len(User[user])
        user_train[user] = [
            (i, ih)
            for i, ih, ts in User[user] if ts <= cutoffs[0]
        ]
        if len(user_train[user]) == 0:
            user_train.pop(user)
        for i_test in range(num_test_groups):
            user_test[i_test][user] = [
                (i, ih)
                for i, ih, ts in User[user]
                if ts <= cutoffs[i_test+1] and ts > cutoffs[i_test]
            ]
            if len(user_test[i_test][user]) == 0:
                user_test[i_test].pop(user)
    # keep only users that have data in all ts groups
    for user in User:
        if not all(
            user in user_test[i_test]
            for i_test in range(num_test_groups)
        ):
            for j_test in range(num_test_groups):
                if user in user_test[j_test]:
                    user_test[j_test].pop(user)
    return [user_train, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # a list of tuple [([itemid], item_history)]
    neg_pool = [x for x in test.values() if len(x) > 0]

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        # join the train seq and the validation element together
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx], _ = valid[u][0]
        idx -= 1
        for i, ih in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set([i for i, _ in train[u]])
        rated.add(0)
        item_idx, pos_ih = [test[u][0][0]], test[u][0][1]
        ih_len = min(args.maxlen_ih, len(pos_ih))
        item_history = [[0] * (args.maxlen_ih - ih_len) + pos_ih[-ih_len:]]
        for _ in range(100):
            t = np.random.randint(0, len(neg_pool))
            while neg_pool[t][0][0] in rated: t = np.random.randint(0, len(neg_pool))
            item_idx.append(neg_pool[t][0][0])
            neg_spl_ih_len = min(args.maxlen_ih, len(neg_pool[t][0][1]))
            neg_spl_ih = [0] * (args.maxlen_ih - neg_spl_ih_len) + neg_pool[t][0][1][-neg_spl_ih_len:]
            item_history.append(neg_spl_ih)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx, item_history]]
        )
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    neg_pool = [x for x in test.values() if len(x) > 0]

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i, _ in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set([i for i, _ in train[u]])
        rated.add(0)
        item_idx, pos_ih = [valid[u][0][0]], valid[u][0][1]
        ih_len = min(args.maxlen_ih, len(pos_ih))
        item_history = [[0] * (args.maxlen_ih - ih_len) + pos_ih[-ih_len:]]
        for _ in range(100):
            t = np.random.randint(0, len(neg_pool))
            while neg_pool[t][0][0] in rated: t = np.random.randint(0, len(neg_pool))
            item_idx.append(neg_pool[t][0][0])
            neg_spl_ih_len = min(args.maxlen_ih, len(neg_pool[t][0][1]))
            neg_spl_ih = [0] * (args.maxlen_ih - neg_spl_ih_len) + neg_pool[t][0][1][-neg_spl_ih_len:]
            item_history.append(neg_spl_ih)
            """
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
            """

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx, item_history]]
        )
        """
        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx]]
        )
        """
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def evaluate_by_ts(model, dataset, ts_group, args):
    [train, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # a list of tuple [([itemid], item_history)]
    # neg_pool = [x for x in test[ts_group].values() if len(x) > 0]
    num_test_groups = 5
    neg_pool = [
        x
        for igroup in range(num_test_groups)
        for x in test[igroup].values() if len(x) > 0
    ]

    users = [u for u, x in test[ts_group].items() if len(x) > 0]
    if usernum>10000:
        users = random.sample(users, 10000)

    print(f"test users in group {ts_group}: {len(users)}")
    for u in users:
        # join the train seq and the validation element together
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i_group in reversed(range(ts_group-1)):
            for i, ih in reversed(test[ts_group].get(u, [])):
                if idx == -1: break
                seq[idx] = i
                idx -= 1
        for i, ih in reversed(train.get(u, [])):
            if idx == -1: break
            seq[idx] = i
            idx -= 1
        rated = set([i for i, _ in train.get(u, [])])
        rated.add(0)
        for i_test in range(len(test[ts_group].get(u, []))):
            item_idx, pos_ih = [test[ts_group][u][i_test][0]], test[ts_group][u][i_test][1]
            ih_len = min(args.maxlen_ih, len(pos_ih))
            item_history = [[0] * (args.maxlen_ih - ih_len) + pos_ih[-ih_len:]]
            for _ in range(100):
                t = np.random.randint(0, len(neg_pool))
                while neg_pool[t][0][0] in rated: t = np.random.randint(0, len(neg_pool))
                item_idx.append(neg_pool[t][0][0])
                neg_spl_ih_len = min(args.maxlen_ih, len(neg_pool[t][0][1]))
                neg_spl_ih = [0] * (args.maxlen_ih - neg_spl_ih_len) + neg_pool[t][0][1][-neg_spl_ih_len:]
                item_history.append(neg_spl_ih)

        predictions = -model.predict(
            *[np.array(l) for l in [[u], [seq], item_idx, item_history]]
        )
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def evaluate_by_ts_degree(model, dataset, ts_group, args):
    [train, test, usernum, itemnum] = copy.deepcopy(dataset)

    def degree_to_degbucket(_item_freq):
        if _item_freq == 0:
            return 0
        elif _item_freq <= 5:
            return 1
        elif _item_freq <= 10:
            return 2
        elif _item_freq <= 100:
            return 3
        return 4

    # a list of tuple [([itemid], item_history)]?
    # neg_pool = [x for x in test[ts_group].values() if len(x) > 0]
    # fix the neg pool for all ts groups
    neg_pool = [
        (item_id, ih)
        for igroup in range(len(test))
        for x in test[igroup].values()
        for item_id, ih in x
    ]

    # frequency of item in training data
    items_train = [x for l in train.values() for x, _ in l]
    values, counts = np.unique(items_train, return_counts=True)
    item_freq = dict(zip(values, counts))

    # break neg pool by degree
    num_degree_buckets = degree_to_degbucket(100000) + 1
    neg_pool_by_degree = {
        degbucket: [
            (x, ih)
            for x, ih in neg_pool
            if degree_to_degbucket(
                item_freq.get(x, 0)
            ) == degbucket
        ]
        for degbucket in range(num_degree_buckets)
    }

    users = [u for u, x in test[ts_group].items() if len(x) > 0]
    if usernum>10000:
        users = random.sample(users, 10000)

    print(f"test users in group {ts_group}: {len(users)}")
    NDCG = {}
    HT = {}
    valid_user = {}
    pool_size = {
        degbucket: len(negpool)
        for degbucket, negpool in neg_pool_by_degree.items()
    }

    n_iter = 0
    for u in users:
        # join the train seq and the validation element together
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        """
        for i_group in reversed(range(ts_group-1)):
            for i, ih in reversed(test[ts_group].get(u, [])):
                if idx == -1: break
                seq[idx] = i
                idx -= 1
        """
        for i, ih in reversed(train.get(u, [])):
            if idx == -1: break
            seq[idx] = i
            idx -= 1
        rated = set([i for i, _ in train.get(u, [])])
        rated.add(0)
        degbucket_cnt_for_user = {degbucket: 0 for degbucket in neg_pool_by_degree.keys()}
        for i_test in range(len(test[ts_group].get(u, []))):
            item_idx, pos_ih = [test[ts_group][u][i_test][0]], test[ts_group][u][i_test][1]
            # identify the degree bucket
            _item_freq = item_freq.get(item_idx[0], 0)
            item_degbucket = degree_to_degbucket(_item_freq)

            ih_len = min(args.maxlen_ih, len(pos_ih))
            item_history = [[0] * (args.maxlen_ih - ih_len) + pos_ih[-ih_len:]]
            # only sample from the neg pool of this bucket
            for _ in range(100):
                t = np.random.randint(0, len(neg_pool_by_degree[item_degbucket]))
                while neg_pool_by_degree[item_degbucket][t][0] in rated:
                    t = np.random.randint(0, len(neg_pool_by_degree[item_degbucket]))
                item_idx.append(neg_pool_by_degree[item_degbucket][t][0])
                neg_spl_ih_len = min(
                    args.maxlen_ih,
                    len(neg_pool_by_degree[item_degbucket][t][1])
                )
                neg_spl_ih = (
                    [0] * (args.maxlen_ih - neg_spl_ih_len) +
                    neg_pool_by_degree[item_degbucket][t][1][-neg_spl_ih_len:]
                )
                item_history.append(neg_spl_ih)

            predictions = -model.predict(
                *[np.array(l) for l in [[u], [seq], item_idx, item_history]]
            )
            predictions = predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user[item_degbucket] = valid_user.setdefault(item_degbucket, 0) + 1

            if rank < 10:
                NDCG[item_degbucket] = NDCG.setdefault(item_degbucket, 0) + 1 / np.log2(rank + 2)
                HT[item_degbucket] = HT.setdefault(item_degbucket, 0) + 1
            if n_iter % 100 == 0:
                print('.', end="")
                sys.stdout.flush()
            n_iter += 1

    for item_degbucket in valid_user.keys():
        NDCG[item_degbucket] = NDCG.get(item_degbucket, 0) * 1.0 / valid_user[item_degbucket]
        HT[item_degbucket] = HT.get(item_degbucket, 0) * 1.0 / valid_user[item_degbucket]

    return NDCG, HT, valid_user, pool_size
