import sys
import torch
import numpy as np
import random
import logging
from transformers import AutoTokenizer, AutoModel, AutoConfig

PATH_TO_SENTEVAL = ''
PATH_TO_DATA = ''
LAYERS_TO_USE = [1,2,3,4,5,6,7,8,9,10,11,12]

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

"""
This is the file to do the senteval probing. It is possible to do multiple layers and seeds.

"""

def prepare(params, samples):
    return


def batcher(params, batch):
    layer_to_use = params['layer_to_use']

    batch = [" ".join(sent) if len(sent) > 0 else "." for sent in batch]

    batch = [["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch]
    batch = [b[:512] for b in batch]

    seq_length = max([len(sent) for sent in batch])
    mask = [[1] * len(sent) + [0] * (seq_length - len(sent)) for sent in batch]
    segment_ids = [[0] * seq_length for _ in batch]

    batch = [tokenizer.convert_tokens_to_ids(sent) + [0] * (seq_length - len(sent)) for sent in batch]

    with torch.no_grad():
        batch = torch.tensor(batch).cuda()
        mask = torch.tensor(mask).cuda()
        segment_ids = torch.tensor(segment_ids).cuda()

        outputs = model(
            batch,
            attention_mask=mask,
            token_type_ids=segment_ids,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states

    extended_mask = mask.unsqueeze(-1)

    embeddings = {}
    for layer in range(1, 13):
        output = hidden_states[layer]
        output = extended_mask * output
        output = torch.sum(output, -2) / torch.sum(mask, -1).unsqueeze(-1)
        embeddings[layer] = output.cpu().numpy()

    return embeddings[layer_to_use]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    model_name = ""

    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    model.eval()
    model.cuda()

    probing_task = ['CoordinationInversion']

    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    seeds = [1,2,3,4,5]
    
    # Dict to collect scores per layer across seeds
    layer_scores = {layer: [] for layer in LAYERS_TO_USE}

    for seed in seeds:
        print(f"\nRunning seed {seed}\n")
        set_seed(seed)
        for layer in LAYERS_TO_USE:
            params = {
                'task_path': PATH_TO_DATA,
                'usepytorch': True,
                'layer_to_use': layer,
                'seed': seed  # ← added
            }
            params['classifier'] = {
                'nhid': 0,
                'optim': 'adam',
                'batch_size': 64,
                'tenacity': 5,
                'epoch_size': 4
            }
            se = senteval.engine.SE(params, batcher, prepare)
            results = se.eval(probing_task)
            score = results['CoordinationInversion']['acc']
            layer_scores[layer].append(score)
            print(f"Seed {seed} | Layer {layer}: {score}")


    print("\n==============================")
    print("FINAL RESULT")
    print("==============================")

    for layer in LAYERS_TO_USE:
        scores = layer_scores[layer]
        print(f"Layer {layer}: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")