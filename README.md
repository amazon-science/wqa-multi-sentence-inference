# WQA Joint Models

This repository contains the code to reproduce the results of the paper: [Paragraph-based Transformer Pretraining for Multi-Sentence Inference](<Update_link_to_arxiv_paper>). The framework is based on the libraries `datasets`, `transformers` and `pytorch-lightning` to be as extensible as possible.


# Index

* [Environment](#environment)
* [Create Dataset](#create-dataset)
* [Continuous Pretraining](#continuous-pretraining)
* [Fine Tuning and Testing](#finetuning-testing)

<a name="environment"></a>
# Environment

Create or activate your conda environment and then install the required libraries:

```bash
conda create --name nlp python==3.9.7
conda activate nlp
make env
```

<a name="create-dataset"></a>
# Create Datasets

## Pretraining

In the original paper we used CC-News, Wikipedia (English), BookCorpus and OpenWebText as datasets. We will show an example of how to transform a pretraining dataset in our format using the `openwebtext`. We use HF datasets for accessing the datasets everywhere, as it is easily publicly available.

```bash
python -m process_datasets \
    --loader DatasetLoader \
    --name openwebtext \
    --output_folder datasets/openwebtext-sentence-sentence-dataset \
    --batch_size 20000 \
    --premise_range 1 1 \
    --consequence_range 1 2 \
    --max_hard_negatives 2 \
    --max_negatives 4 \
    --paragraph_ratio 1.0 \
    --document_ratio 1.0 \
    --num_proc 32 \
    --field text \
    --paragraph_separator '\n\n'
```

Another example with the English Wikipedia is the following:

```bash
python -m process_datasets \
    --loader DatasetLoader \
    --name wikipedia:20200501.en \
    --output_folder datasets/wikipediaen-sentence-sentence-dataset \
    --batch_size 20000 \
    --premise_range 1 1 \
    --consequence_range 1 2 \
    --max_hard_negatives 2 \
    --max_negatives 4 \
    --paragraph_ratio 1.0 \
    --document_ratio 1.0 \
    --num_proc 32 \
    --field text \
    --paragraph_separator '\n\n'
```

A few words about CC-News and BookCorpus:
    - Please do not use the `cc_news` dataset from the HF repository because it is a very reduced version of the original. If you need to run experiments with this, please contact us.
    - We suggest to use the `bookcorpusopen` from HF datasets instead of `bookcorpus` because it is larger and was not preprocessed.


### Arguments

* `--loader DatasetLoader` is the class that will load the dataset from the HF repository. If you have a dataset dump (saved with `dataset.save_to_disk()`) you should use `--loader DiskDataset` while if you have a dataset as `jsonl` files you should use `--loader JsonLoader`.
* `--batch_size` is the number of input documents that are processed together, so the negatives will be created all within that batch, than the larger the better. However, notice that a very large `batch_size` will require a large amout of RAM. 
* `--premise_range` and `--consequence_range` define the range (in terms of number of sentences) of the left (question) and right (candidates) part of the input examples. The lengths will be randomly sampled from this range for every example.
* `--max_hard_negatives` defines the maximum number of negatives that should be created from within the same document (but different paragraph).
* `--max_negatives` is instead the total number of negatives desired, and will take negatives from other documents until the total is the desired number.
* `--paragraph_ratio` and `--document_ratio` can be used to control the fraction of documents that are used and the fraction of paragraphs for each document to consider. This can be useful to reduce the final size of the dataset. Excluded paragraphs and documents can still be used to create negatives.
* `--num_proc` is the number of CPUs to use.
* `--field` is the field of each input example that should be considered. For example, `wikipedia` HF dumps store the text in a field named `maintext`, so you should set `--field maintext`.
* `--paragraph_separator` is the string that will be used to split a document in many paragraphs. Most of the datasets use the double newline but there are some exceptions like CC-News, which uses just a single newline.

The output will be a datasets instance which can be easily loaded and explored with
```python
import datasets
owt = datasets.load_from_disk('datasets/openwebtext-sentence-sentence-dataset')
```

## Finetuning

We do not provide any links to directly download the processed datasets, but rather provide the instructions to build WikiQA, TREC-QA, ASNQ and FEVER from scratch in a few steps.

### AS2 Datasets

Since some experiments needs to select the best `k` candidates, we are providing a version of each AS2 dataset with a column containing the scores computed by a Pointwise RoBERTa-Base model fine-tuned on the same data. For ASNQ, the scripts will automatically download the splits released [here](https://github.com/alexa/wqa-cascade-transformers) to have both a dev and a test set.

To create all the AS2 datasets, just run:

```bash
make wikiqa
make trecqa
make asnq
```

### Fact Checking

Note: the FEVER dataset does not have labels in the test set. You may need to submit predictions to the leaderboard to check test performance.

```bash
make fever
```


<a name="continuous-pretraining"></a>
# Continuous Pretraining

The pretraining of the two models can be done with the following script, just change `--head_type` to `IE_k` or `AE_k` to use a different head type. `IE_k` uses just the representation of every candidate to make `k` predictions while `AE_k` uses the concatenation of the representation of the question and every candidate to make the predictions. More details can be found in the paper.

The following is an example of pretraining script with `IE_k` as classification head. We also keep MLM because we found that combining the losses improves the final performance.

```bash
python -m transformers_framework \
    --model RobertaJointMLMAndClassification \
    --devices 8 --accelerator gpu --strategy ddp \
    --precision 16 \
    --pre_trained_model roberta-base \
    --name roberta-base-joint-sentence-sentence-IE-k \
    --output_dir outputs/joint-pretraining \
    \
    --adapter JointwiseArrowAdapter \
    --batch_size 64 \
    --train_filepath \
        datasets/openwebtext-sentence-sentence-dataset \
        datasets/wikipediaen-sentence-sentence-dataset \
    --field_names premise consequence \
    --label_name label \
    \
    --log_every_n_steps 1000 \
    --accumulate_grad_batches 8 \
    --max_sequence_length 64 \
    -k 5 --selection random --separated \
    --learning_rate 1e-04 \
    --max_steps 100000 \
    --weight_decay 0.01 \
    --num_warmup_steps 10000 \
    --num_workers 8 \
    --head_type IE_k \
    --seed 1337
```

### Arguments

* `--model` the model class to use. `RobertaJointMLMAndClassification` implements MLM and multi-candidate classification.
* `--devices 8 --accelerator gpu --strategy ddp ` tells the system to use 8 GPUs with PyTorch's `ddp`. If you want to use `deepspeed` to save memory, just install it with `pip install deepspeed` and set `--strategy deepspeed_stage_2`.
* `--precision` tells the system to use full (`32`) or half (`16`) precision. 
* `--name` a custom name for the run, will be used to differentiate between the pretrained models in the output folder.
* `--output_dir` pretrained models, checkpoints and tensorboard will be save here.
* `--adapter` the data loading adapter to use.
* `--batch_size` training batch size per device. Total batch size is `batch_size * devices * accumulation`.
* `--train_filepath` path to the datasets in HF datasets format.
* `--accumulate_grad_batches` for gradient accumulation.
* `--max_sequence_length` for each question/candidates. Total sequence length is `max_sequence_length * (k + 1)`.
* `-k 5 --selection random --separated` candidates will be randomly divided in groups of `5` and internal padding will be used to obtain an even separation.
* `--learning_rate 1e-04 --max_steps 100000 --weight_decay 0.01 --num_warmup_steps 10000` will train with a peak LR of `1e-04` for `100K` steps and with weight decay. The LR will be warmed up in the first `10K` steps and then linearly decreased.
* `--num_workers` number of dataloader workers.
* `--head_type` which joint classification head to use. Choose between `IE_k` and `AE_k` for pretraining.
* `--seed 1337` for reproducibility.


The script are also available in the folder `transformers_experiments/pretraining`.


For those who would like to have the already pretrained models on BookCorpus, CC-News, OpenWebText and Wikipedia (English), we are resealsing them:
* [Joint model pretrained with IE_k head.](http://)
* [Joint model pretrained with AE_k head.](http://)

**NEWS**: More models trained on better data and for a longer time will arrive soon.


<a name="finetuning-testing"></a>
# Fine Tuning and Testing

## AS2

To launch a finetuning, it is enough to run one of the scripts in the folder `transformers_experiments/finetuning`. For example, to finetune a pretrained model on `ASNQ` with the same hyperparameters of the paper you may run the following:

```bash
python -m transformers_framework \
    --model RobertaJointAS2 \
    --devices 8 --accelerator gpu --strategy ddp \
    --precision 16 \
    --pre_trained_model <path-to-pretrained-model> \
    --name roberta-base-joint-sentence-sentence-IE-k \
    --output_dir outputs/joint-asnq \
    \
    --adapter JointwiseArrowAdapter \
    --batch_size 128 --val_batch_size 128 --test_batch_size 128 \
    --train_filepath datasets/asnq --train_split train \
    --valid_filepath datasets/asnq --valid_split validation \
    --test_filepath datasets/asnq --test_split test \
    --field_names question answer \
    --label_name label \
    --key_name key \
    \
    --accumulate_grad_batches 2 \
    --max_sequence_length 64 \
    -k 5 --selection all --force_load_dataset_in_memory --separated \
    --learning_rate 1e-05 \
    --max_epochs 6 \
    --early_stopping \
    --patience 6 \
    --weight_decay 0.0 \
    --num_warmup_steps 5000 \
    --monitor validation/map \
    --val_check_interval 0.5 \
    --num_workers 8 \
    --head_type IE_k \
    --seed 1337
```

The training arguments are mostly as before. `-k 5 --selection all` uses all available candidates and divides them in groups of `k` (the last group may contain less candidates). Since the creation of groups is done once when the dataset is instantiated, they may be the same for all the epochs. To create new groups at every epoch, `--reload_dataloaders_every_n_epoch 1` tells the system to reload the dataset after every epoch (and thus create new groups) while `--shuffle_candidates` make sure candidates are shuffled within a single group. Finally, `--force_load_dataset_in_memory` is needed when `--selection all` because the number of rows of the original dataset may be different from the number of rows after splitting the candidates in groups of `k`, so we load everything in memory to know the final number of candidates before starting the training.

If you want to train only on the best (or worst) `k` candidates, you should set `--selection best` and tell the system where to take the scores with `--score_name scores_roberta_base`. All the AS2 datasets we provide already have the `scores_roberta_base` column inside.

The script will automatically run a validation on the dev set two times per epoch (`--val_check_interval 0.5`) and will stop if the Mean Average Precision (`--monitor validation/map`) does not improve for 6 consecutive validations (`--patience 6`). Finally, it will test the best checkpoint and print the results on screen and to the tensorboards.

To obtain the results of the baseline joint models, you may just run the scripts inside `transformers_experiments/finetuning/*` setting `--pre_trained_model roberta-base`. For the results with the pretrained joint models, just set `--pre_trained_models <path-to-pretrained-model>` passing the models downloaded before.


## Fact Checking

An example of script to run a finetuning is the following:

```bash
python -m transformers_framework \
    --model RobertaJointFactChecking \
    --devices 2 --accelerator gpu --strategy ddp \
    --pre_trained_model <path-to-pretrained-model> \
    --name roberta-base-joint-fever-AE-1 \
    --output_dir outputs/joint-fever \
    \
    --adapter JointwiseArrowAdapter \
    --batch_size 32 --val_batch_size 128 --test_batch_size 128 \
    --train_filepath datasets/fever --train_split train \
    --valid_filepath datasets/fever --valid_split validation \
    --field_names claim evidence \
    --label_name label \
    --key_name key \
    \
    --accumulate_grad_batches 1 \
    --max_sequence_length 64 \
    -k 5 --selection all --separated --force_load_dataset_in_memory --reduce_labels \
    --learning_rate 1e-05 \
    --max_epochs 15 \
    --early_stopping \
    --patience 8 \
    --weight_decay 0.0 \
    --num_warmup_steps 1000 \
    --monitor validation/accuracy \
    --val_check_interval 0.5 \
    --num_workers 8 \
    --shuffle_candidates --reload_dataloaders_every_n_epoch 1 \
    --num_labels 3 \
    --head_type AE_1 \
```

Since the FEVER dataset does not have labels for the test split, you may just generate predictions after the finetuning with:

```bash
python -m transformers_framework \
    --model RobertaJointFactChecking \
    --devices 1 --accelerator gpu \
    --pre_trained_model <path-to-pretrained-and-finetuned-model> \
    --name roberta-base-joint-fever-AE-1-test \
    --output_dir outputs/joint-fever \
    \
    --adapter JointwiseArrowAdapter \
    --predict_batch_size 128 \
    --predict_filepath datasets/fever --predict_split test \
    --field_names claim evidence \
    --label_name label \
    --key_name key \
    \
    --max_sequence_length 64 \
    -k 5 --selection all --separated --force_load_dataset_in_memory --reduce_labels \
    --num_workers 8 \
    --shuffle_candidates --reload_dataloaders_every_n_epoch 1 \
    --num_labels 3 \
    --head_type AE_1
```


# FAQ

* The datasets takes forever to load &#8594; move it on the main device of your machine.
* I have a single GPU &#8594; just set `--devices 1 --accelerator gpu` without `--strategy`.
* I want to pretrain on many machines &#8594; set `--num_nodes X` with `X` equal to the number of machines and launch the script a single time for every machine with the following environment variables: `MASTER_ADDR=<ip-main-node>`, `MASTER_PORT=<some-free-port-on-main-node>`, `NODE_RANK=<0...X-1>`, `WORLD_SIZE=X`.
* Need the datasets links? You can find them in the `Makefile`.
* The core training algorithm is based on `pytorch-lightning`, so you may use every parameter described [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html).
* The output folder contains up to `4` different folders called `tensorboard`, `pre_trained_models`, `checkpoints` and `predictions`:
    * `tensorboard` contains the logs for every experiment that you run;
    * `pre_trained_models` contains just the models checkpoints (saved with `model.save_pretrained()`) only of the models;
    * `checkpoints` contains the `pytorch-lightning` checkpoints instead, which includes also the dump of all the optimizers and scheduler states. This checkpoints should be used just to restore a previously running training session.
    * Finally, the `predictions` folder appears only if you do predictions on some dataset.


# Citation

Please cite our work if you find this repository useful.
```bibtex
@misc{https://doi.org/10.48550/arxiv.2205.01228,
  doi = {10.48550/ARXIV.2205.01228},
  
  url = {https://arxiv.org/abs/2205.01228},
  
  author = {Di Liello, Luca and Garg, Siddhant and Soldaini, Luca and Moschitti, Alessandro},
  
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Paragraph-based Transformer Pre-training for Multi-Sentence Inference},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```

# Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

# License

This library is licensed under the CC-BY-NC-4.0 License.
