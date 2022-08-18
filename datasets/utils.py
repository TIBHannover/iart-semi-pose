import logging
import torch
import json
import torch.distributed as dist


def split_dict(data_dict, splits):
    data = flat_dict(data_dict)
    result = [{}] * len(splits)
    for k, v in data.items():
        result_list = torch.split(v, splits)
        for i in range(len(splits)):
            result[i][k] = result_list[i]

    return [unflat_dict(x) for x in result]


def unflat_dict(data_dict, parse_json=False):
    result_map = {}
    if parse_json:
        data_dict_new = {}
        for k, v in data_dict.items():
            try:
                data = json.loads(v)
                data_dict_new[k] = data
            except:
                data_dict_new[k] = v
        data_dict = data_dict_new
    for k, v in data_dict.items():
        path = k.split(".")
        prev = result_map
        for p in path[:-1]:
            if p not in prev:
                prev[p] = {}
            prev = prev[p]
        prev[path[-1]] = v
    return result_map


def flat_dict(data_dict, parse_json=False):
    result_map = {}
    for k, v in data_dict.items():
        if isinstance(v, dict):
            embedded = flat_dict(v)
            for s_k, s_v in embedded.items():
                s_k = f"{k}.{s_k}"
                if s_k in result_map:
                    logging.error(f"flat_dict: {s_k} alread exist in output dict")

                result_map[s_k] = s_v
            continue

        if k not in result_map:
            result_map[k] = []
        result_map[k] = v
    return result_map


def get_element(data_dict: dict, path: str, split_element="."):
    if path is None:
        return data_dict

    if callable(path):
        elem = path(data_dict)

    if isinstance(path, str):
        elem = data_dict
        try:
            for x in path.strip(split_element).split(split_element):
                try:
                    x = int(x)
                    elem = elem[x]
                except ValueError:
                    elem = elem.get(x)
        except:
            pass

    if isinstance(path, (list, set)):
        elem = [get_element(data_dict, x) for x in path]

    return elem


def read_jsonl(path, dict_key=None, keep_keys=None):
    data = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            if keep_keys is not None:
                d = {k: get_element(d, v) for k, v in keep_keys.items()}
            data.append(d)

    if dict_key is not None:
        data = {get_element(x, dict_key): x for x in data}

    return data


def write_jsonl_lb_mapping(path, outpath):
    # read icon meta file and write the mapping file
    data = read_jsonl(path)
    with open(outpath, mode="a") as f:
        for i in data:
            json_rec = json.dumps({i["id"]: i["kw"]})
            f.write(json_rec + "\n")


def read_jsonl_lb_mapping(path):
    # read label mapping file
    # output a dictionary key:id value:labels
    data = dict()
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            data.update(d)

    return data
