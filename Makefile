# Code check and install requirements
check:
	# isort and flake
	isort .
	flake8 .

env:
	pip install -r requirements.txt

wikiqa:
	rm -rf datasets/wikiqa /tmp/wikiqa_tmp
	python transformers_utilities/datasets/create_wikiqa_dataset.py --output_folder /tmp/wikiqa_tmp
	python transformers_utilities/datasets/merge_datasets.py \
		--input /tmp/wikiqa_tmp datasets/scores_as2/scores_roberta_base_wikiqa \
		--output datasets/wikiqa
	rm -r /tmp/wikiqa_tmp

trecqa:
	rm -rf datasets/trecqa /tmp/lexdecomp-master /tmp/trecqa.zip /tmp/trecqa_tmp
	wget https://github.com/mcrisc/lexdecomp/archive/refs/heads/master.zip -O /tmp/trecqa.zip
	unzip /tmp/trecqa.zip -d /tmp
	python transformers_utilities/datasets/create_trecqa_dataset.py \
		--input_folder /tmp/lexdecomp-master/trec-qa \
		--output_folder /tmp/trecqa_tmp
	python transformers_utilities/datasets/merge_datasets.py \
		--input /tmp/trecqa_tmp datasets/scores_as2/scores_roberta_base_trecqa \
		--output datasets/trecqa
	rm -r /tmp/lexdecomp-master /tmp/trecqa.zip /tmp/trecqa_tmp

asnq:
	rm -rf datasets/asnq /tmp/asnq.tar /tmp/data /tmp/wqa-cascade-transformers-master /tmp/asnq_tmp
	wget https://d3t7erp6ge410c.cloudfront.net/tanda-aaai-2020/data/asnq.tar -O /tmp/asnq.tar
	tar xvf /tmp/asnq.tar -C /tmp
	wget https://github.com/alexa/wqa-cascade-transformers/archive/refs/heads/master.zip -O /tmp/cascade.zip
	unzip /tmp/cascade.zip -d /tmp
	python transformers_utilities/datasets/create_asnq_dataset.py \
		--input_folder /tmp/data/asnq \
		--output /tmp/asnq_tmp \
		--dev_filter /tmp/wqa-cascade-transformers-master/acl2020cascade/data/unique.dev \
		--test_filter /tmp/wqa-cascade-transformers-master/acl2020cascade/data/unique.test
	python transformers_utilities/datasets/merge_datasets.py \
		--input /tmp/asnq_tmp datasets/scores_as2/scores_roberta_base_asnq \
		--output datasets/asnq
	rm -r /tmp/cascade.zip /tmp/asnq.tar /tmp/data /tmp/wqa-cascade-transformers-master /tmp/asnq_tmp

fever:
	rm -rf datasets/fever /tmp/kgat.zip /tmp/KernelGAT
	wget https://thunlp.oss-cn-qingdao.aliyuncs.com/KernelGAT/FEVER/KernelGAT.zip -O /tmp/kgat.zip
	unzip /tmp/kgat.zip -d /tmp
	python transformers_utilities/datasets/create_fever_dataset.py \
		--train_file /tmp/KernelGAT/data/bert_train.json \
		--dev_file /tmp/KernelGAT/data/bert_dev.json \
		--test_file /tmp/KernelGAT/data/bert_test.json \
		--output datasets/fever
	rm -r /tmp/kgat.zip /tmp/KernelGAT
