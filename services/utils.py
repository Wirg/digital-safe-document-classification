def n_closest(from_, compare_to, similarity, n=5):
    return similarity(compare_to, from_)[:, 0].argsort()[::-1][:n]
