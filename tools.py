import numpy as np
import pandas as pd



def normalize_title(title):
    normalized_title = ""
    title = title.strip()
    for c in title:
        if c == '#':
            break
        normalized_title += c
    return normalized_title


def get_intersection(wallet_collections, top_collections):
    top_list = list(top_collections)
    inter = np.full(len(top_list), 0)
    for i in range(0, len(top_list)):
        inter[i] = top_list[i] in wallet_collections
        # inter[i] = max(1, i % 3)

    df = pd.DataFrame(inter).T
    df.columns = top_list
    return df
