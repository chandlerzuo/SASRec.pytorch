from collections import defaultdict

raw_file = "ratings.dat"
processed_file = "ml-1m.txt"
max_ih_len = 30
# processed_file = "ml-1m_full.txt"
min_degree = 5


def parse(path):
    g = open(path, "r")
    for l0 in g:
        yield l0.strip().split("::")


countU = defaultdict(lambda: 0)
countP = defaultdict(lambda: 0)
line = 0

for l in parse(raw_file):
    line += 1
    rev, asin, _, _ = l
    countU[rev] += 1
    countP[asin] += 1

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
for user in User.keys():
    for timestamp, itemid in User[user]:
        item_history = []
        for x in Item[itemid]:
            if x[0] >= timestamp:
                break
            item_history.append(x[1])
        if len(item_history) > max_ih_len:
            item_history = item_history[-max_ih_len:]
        ih_str = ",".join([str(x) for x in item_history])
        f.write(f"{user} {itemid} {timestamp} {ih_str}\n")
    nrow += 1
    if nrow % 200 == 0:
        print(f"processed {nrow} users")
f.close()
