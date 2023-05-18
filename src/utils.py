import json


def make_dict_from_json(path: str):
    with open(path, "r") as f:
        data = json.loads(f.read())
    return data


id2label = make_dict_from_json("../hackathon/id2label_final.json")
id2label = {k: int(v) for k, v in id2label.items()}
label2id = make_dict_from_json("../hackathon/label2id_final.json")
label2id = {int(k): v for k, v in label2id.items()}

if __name__ == "__main__":
    print(id2label, label2id)
