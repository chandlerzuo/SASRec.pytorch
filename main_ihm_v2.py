import argparse
import os
import time

import numpy as np

import torch

from .model_ihm_v2 import SASRecIHM as SASRec

# from model import SASRec
from .utils_ihm import (
    data_partition_by_ts,
    evaluate,
    evaluate_by_ts_degree_v2,
    WarpSampler,
)

# instructions https://www.internalfb.com/intern/wiki/Oculus-nimble/Building_&_Testing_Nimble_code/modifying-torch-code/
# buck run @mode/opt //sasrec:main_ihm_v2 -- --dataset=ml-1m_full --data_dir=ml-1m_full --train_dir=ml-1m_full_v2_ihm --maxlen=200 --dropout_rate=0.2 --num_epochs 100 --model IHM --num_experts=2 --test=true

# python3 main_ihm.py --dataset=amazon_beauty --data_dir=amazon --train_dir=beauty_ts_group_negpool_bucket_base --maxlen=200 --dropout_rate=0.2 --num_epochs 100 --model SASRec


def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--data_dir", required=True, type=str)
parser.add_argument("--train_dir", required=True)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=50, type=int)
parser.add_argument("--maxlen_ih", default=10, type=int)
parser.add_argument("--hidden_units", default=50, type=int)
parser.add_argument("--num_blocks", default=2, type=int)
parser.add_argument("--num_epochs", default=201, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.5, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cpu", type=str)
parser.add_argument("--inference_only", default=False, type=str2bool)
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--model", default=None, type=str)
parser.add_argument("--gamma", default=0, type=float)
parser.add_argument("--eval_every_n_epoch", default=20, type=int)
parser.add_argument(
    "--test", default=False, type=str2bool
)  # only train 1 batch, and eval after every epoch
parser.add_argument(
    "--raw_loss_weight", default=1.0, type=float
)  # weight on the raw SASRec embedding loss
parser.add_argument("--num_experts", default=1, type=int)  # num of IHM experts

args = parser.parse_args()
if not os.path.isdir("sasrec/experiments/" + args.dataset + "_" + args.train_dir):
    os.makedirs("sasrec/experiments/" + args.dataset + "_" + args.train_dir)
with open(
    os.path.join(
        "sasrec/experiments/" + args.dataset + "_" + args.train_dir, "args.txt"
    ),
    "w",
) as f:
    f.write(
        "\n".join(
            [
                str(k) + "," + str(v)
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])
            ]
        )
    )
f.close()

if __name__ == "__main__":
    # global dataset
    # dataset = data_partition(args.dataset)
    dataset = data_partition_by_ts(args.data_dir, args.dataset)

    # [user_train, user_valid, user_test, usernum, itemnum] = dataset
    [user_train, _, usernum, itemnum] = dataset
    num_batch = (
        len(user_train) // args.batch_size
    )  # tail? + ((len(user_train) % args.batch_size) != 0)
    if args.test:
        num_batch = 1
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    f = open(
        os.path.join(
            "sasrec/experiments/" + args.dataset + "_" + args.train_dir, "log.txt"
        ),
        "w",
    )

    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        maxlen_ih=args.maxlen_ih,
        n_workers=3,
    )
    model = SASRec(usernum, itemnum, args).to(
        args.device
    )  # no ReLU activation in original SASRec implementation?

    for _, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers
    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train()  # enable model training

    epoch_start_idx = 1
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(
                torch.load(args.state_dict_path, map_location=torch.device(args.device))
            )
            tail = args.state_dict_path[args.state_dict_path.find("epoch=") + 6 :]
            epoch_start_idx = int(tail[: tail.find(".")]) + 1
        except:  # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
            print("failed loading state_dicts, pls check file path: ", end="")
            print(args.state_dict_path)
            print(
                "pdb enabled for your quick check, pls type exit() if you do not need it"
            )
            import pdb

            pdb.set_trace()
    if args.inference_only:
        # doesn't work for ts groups
        model.eval()
        t_test = evaluate(model, dataset, args)
        print("test (NDCG@10: %.4f, HR@10: %.4f)" % (t_test[0], t_test[1]))
    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    # bce_criterion = torch.nn.BCEWithLogitsLoss()  # torch.nn.BCELoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()

    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        if args.inference_only:
            break  # just to decrease identition
        for step in range(
            num_batch
        ):  # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            (
                u,
                seq,
                pos,
                neg,
                pos_ih,
                neg_ih,
                weights,
            ) = sampler.next_batch()  # tuples to ndarray
            u, seq, pos, neg, pos_ih, neg_ih, weights = (
                np.array(u),
                np.array(seq),
                np.array(pos),
                np.array(neg),
                np.array(pos_ih),
                np.array(neg_ih),
                np.array(weights),
            )
            pos_logits, neg_logits, pos_logits_raw, neg_logits_raw = model(
                u, seq, pos, neg, pos_ih, neg_ih
            )
            # pos_logits, neg_logits = model(u, seq, pos, neg)
            pos_labels, neg_labels = torch.ones(
                pos_logits.shape, device=args.device
            ), torch.zeros(neg_logits.shape, device=args.device)
            # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
            adam_optimizer.zero_grad()
            indices = np.where(pos != 0)

            # weight processing using CDN
            weights = np.ones(pos_logits.shape, dtype=np.float32)
            if args.gamma > 0:  # using CDN
                assert args.gamma >= 1.0, "gamma parameter has to > 1."
                weights = weights / np.mean(weights)
                from_raw_distribution = np.random.choice(
                    [True, False], weights.shape, replace=True
                )
                alpha = 1 - (epoch * 1.0 / args.num_epochs / args.gamma) ** 2.0
                weights[from_raw_distribution] = alpha
                weights[~from_raw_distribution] = (1 - alpha) * weights[
                    ~from_raw_distribution
                ]
            weights = torch.from_numpy(weights).to(args.device)

            bce_criterion = torch.nn.BCEWithLogitsLoss(weight=weights[indices])
            loss = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss += bce_criterion(neg_logits[indices], neg_labels[indices])
            loss += args.raw_loss_weight * bce_criterion(
                pos_logits_raw[indices], pos_labels[indices]
            )
            loss += args.raw_loss_weight * bce_criterion(
                neg_logits_raw[indices], neg_labels[indices]
            )
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()
            adam_optimizer.step()
            print(
                "loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())
            )  # expected 0.4~0.6 after init few epochs
        if epoch % args.eval_every_n_epoch == 0 or args.test:  # to remove
            model.eval()
            t1 = time.time() - t0
            T += t1
            print("Evaluating ", end="")
            """
            t_test = evaluate(model, dataset, args)
            t_valid = evaluate_valid(model, dataset, args)
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)'
                    % (epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            """
            ts_groups = 5
            print("epoch:%d, time: %f(s)" % (epoch, T))
            for i_group in range(ts_groups):
                ndcg, hrate, valid_users, pool_sizes = evaluate_by_ts_degree_v2(
                    model, dataset, i_group, args
                )
                for train_degree in sorted(valid_users.keys()):
                    for k in ndcg.keys():
                        print(
                            "TS window %d degree %d top k %d (NDCG@10: %.4f, HR@10: %.4f)"
                            % (
                                i_group,
                                train_degree,
                                k,
                                ndcg[k][train_degree],
                                hrate[k][train_degree],
                            )
                        )
                        f.write(
                            f"TS window {i_group} degree {train_degree} top k {k}"
                            f"NDCG {ndcg[k][train_degree]} HRate {hrate[k][train_degree]} "
                            f"valid users {valid_users[train_degree]} "
                            f"neg pool size {pool_sizes[train_degree]}\n"
                        )
            f.flush()
            t0 = time.time()
            model.train()
        if epoch == args.num_epochs:
            folder = "sasrec/experiments/" + args.dataset + "_" + args.train_dir
            fname = "SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth"
            fname = fname.format(
                args.num_epochs,
                args.lr,
                args.num_blocks,
                args.num_heads,
                args.hidden_units,
                args.maxlen,
            )
            torch.save(model.state_dict(), os.path.join(folder, fname))
    f.close()
    sampler.close()
    print("Done")
