import json
import os
import sys
dataset_path = sys.argv[1]

with open(os.path.join(dataset_path, "queries.json"), 'r') as f:
    data = json.load(f)

queries, corpus = [], []

for i, item in enumerate(data):
    queries.append(dict(
        qid=f"0:{i}",
        query_txt="Find a document image that matches the given query.\n" + item["query"],
        query_img_path=None,
        query_modality="text",
        pos_cand_list=[f"0:{i}"],
        neg_cand_list=[],
        task_id=0,
    ))
    corpus.append(dict(
        did=f"0:{i}",
        txt=None,
        img_path=item["saved_filename"],
        modality="image",
    ))

with open(os.path.join(dataset_path, "query.jsonl"), 'w') as f:
    for item in queries:
        f.write(json.dumps(item) + '\n')
        
with open(os.path.join(dataset_path, "cand_pool.jsonl"), 'w') as f:
    for item in corpus:
        f.write(json.dumps(item) + '\n')