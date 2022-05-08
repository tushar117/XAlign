from collections import defaultdict
import random
import json


def get_balanced_data(dataset, total_count, logger):
    balance_count = total_count//3
    count = defaultdict(lambda: 0)
    random.shuffle(dataset)
    final_dataset = []
    for x in dataset:
        if (balance_count - count[x['label']])>0:
            final_dataset.append(x)
            count[x['label']]+=1
    logger.debug("dataset distribution :")
    for k, v in count.items():
        logger.debug("%s %s" % (k, v))
    random.shuffle(final_dataset)
    return final_dataset

def load_jsonl(file_path):
    res = []
    with open(file_path) as dfile:
        for line in dfile.readlines():
            res.append(json.loads(line.strip()))
    return res