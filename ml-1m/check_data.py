def histogram(in_file, out_file):
    with open(in_file, "r") as f:
        lines = f.readlines()
        user_hist = {}
        for line in lines:
            userid = int(line.split(" ")[0])
            if userid not in user_hist:
                user_hist[userid] = 0
            user_hist[userid] += 1

    user_hist = sorted(user_hist.items(), key=lambda x: x[1])

    with open(out_file, "w") as f:
        for x in user_hist:
            f.write("%d,%d\n" % (x[0], x[1]))

histogram("ml-1m.txt", "ml-1m_hist.txt")
histogram("../data/ml-1m.txt", "ml-1m_data_hist.txt")
