# GDMTL

This repository hosts source code for [Generalizing Discriminative Retrieval
Models using Generative Tasks](https://doi.org/10.1145/3442381.3449863)
published in The Web Conference 2021.

The code heavily relies on [PyTorch
Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and [Hugging
Face's Transformers](https://github.com/huggingface/transformers).

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is a
training framework which takes cares of the boilerplate such as the training
loop, and the code structure is largely shaped by this framework. The proposed
models are implemented on top of pretrained BERT and BART from
[huggingface](https://github.com/huggingface/transformers).

## Requirements

```
pip install -r requirements.txt
pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip install -e .
```

## Training

1. Download MSMARCO collection, queries and qrels.

	* [DeepCT](https://github.com/AdeDZY/DeepCT) enriched passages are used for first stage BM25 retrieval.
	* The original passages are used for second stage reranking.

    ```
    mkdir collection/
    cd collection/
	curl -LO http://boston.lti.cs.cmu.edu/appendices/arXiv2019-DeepCT-Zhuyun-Dai/weighted_documents/sqrt_sample_100_keepall_jsonl.zip
    curl -LO https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
    ```

2. Generate training/validation/testing runs.

	* Refer to Anserini
      [msmarco](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)
      page for usage.

	* Once you have the run files, binarize them for efficient load/save. The
      scripts used are from [irtools](https://github.com/binshengliu/irtools).

	```
	cat train.small.sqrt.k1=27-b=0.62.ans_run | cut -f 1,2 | label.py --qrels qrels.train.tsv --append-missing-relevant | binarize.py -o train.small.sqrt.k1=27-b=0.62.ans_run.labeled.npy
	cat dev.small.sqrt.k1=27-b=0.62.ans_run | cut -f 1,2 | label.py --qrels qrels.dev.small.tsv | binarize.py -o dev.small.sqrt.k1=27-b=0.62.ans_run.labeled.npy
	cat eval.small.sqrt.k1=27-b=0.62.ans_run | cut -f 1,2 | label.py | binarize.py -o eval.small.sqrt.k1=27-b=0.62.ans_run.labeled.npy
	cat dl2019-test-small-k1=27-b=0.62.ans_run | cut -f 1,2 | label.py --qrels 2019qrels-pass.txt | binarize.py -o dl2019-test-small-k1=27-b=0.62.ans_run.labeled.npy
	```

3. Update [configurations](conf/gdmtl).

    Paths you need to update:
    ```
    collection
    train_data
    train_query
    valid_data
    valid_qrels
    valid_query
    test_data
    test_query
    ```

    Important hyper parameters:
    ```
	train_bsz
	lr
	min_lr
	accumulate_grad_batches
    ```

4. Run training scripts.

    ```
    env CUDA_VISIBLE_DEVICES=0,1 scripts/train_bert_stl.sh
    env CUDA_VISIBLE_DEVICES=0,1 scripts/train_bert_mtl.sh
    env CUDA_VISIBLE_DEVICES=0,1 scripts/train_bert_mtl3.sh
    env CUDA_VISIBLE_DEVICES=0,1 scripts/train_bart_stl.sh
    env CUDA_VISIBLE_DEVICES=0,1 scripts/train_bart_mtl.sh
    ```

## Inference

```
export CKPT=/path/to/saved/checkpoint

export TEST_QUERY=msmarco-test2019-queries-small.tsv
export TEST_RUN=dl2019-test-small-k1=27-b=0.62.ans_run.labeled.npy

env CUDA_VISIBLE_DEVICES=0,1 python mtl_rank/main.py \
    desc=inference \
    conf=bert-base-stl \
    load_checkpoint=$CKPT \
    eval_mode=ranker \
    test_only=True \
    test_query=$TEST_QUERY \
    test_data=$TEST_RUN
```

## Tips

1. Be careful with updates of [model
   configurations](https://huggingface.co/facebook/bart-base/commits/main). They
   could impact model performance.
2. [Apex](https://github.com/NVIDIA/apex) always works better for me than
   [native fp16](https://pytorch.org/docs/stable/amp.html), in terms of both
   effectiveness and efficiency.
