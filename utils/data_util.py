import random


def split(full_list, split_size=0.2, shuffle=True, random_state=None):
    random.seed(random_state)
    """
        划分数组
        test_size 比例或者数量
    """
    n_total = len(full_list)
    offset = int(n_total * split_size)
    if n_total <= 0 or offset < 1:
        return [], full_list
    if isinstance(split_size, int):
        offset = split_size
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2
