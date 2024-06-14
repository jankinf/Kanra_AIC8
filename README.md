# Data-Centric Robust Learning on ML Models

Kanra team (Zihao Wang, Zhengwei Fang)

Top1 winner of the [AAAI2022 Security AI Challenger VIII](https://tianchi.aliyun.com/competition/entrance/531939/introduction).

### Overview:
Current machine learning competitions mostly seek for a high-performance model given a fixed dataset, while recent Data-Centric AI Competition (https://https-deeplearning-ai.github.io/data-centric-comp/) changes the traditional format and aims to improve a dataset given a fixed model. Similarly, in the aspect of robust learning, many defensive methods have been proposed of deep learning models for mitigating the potential threat of adversarial examples, but most of them strive for a high-performance model in fixed constraints and datasets. Thus how to construct a dataset that is universal and effective for the training of robust models has not been extensively explored.

### Installing Dependencies

- Create conda virtual environment.

	```
	conda create -n stage2 python=3.7 -y
	conda activate stage2
	```

- Install PyTorch and torchvision.

	```
	conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch
	```

- Install [imagecorruptions](https://github.com/bethgelab/imagecorruptions).

	```
	pip install imagecorruptions
	```

- Install [auto-attack](https://github.com/fra31/auto-attack/tree/master).

	```
	pip install git+https://github.com/fra31/auto-attack
	```

- Install [robustbench](https://github.com/RobustBench/robustbench).

	```
	pip install git+https://github.com/RobustBench/robustbench.git@v1.0
	```
	
### Generate corruptions

```
python gen_cor.py
```


### Generate Adversarial Examples

1. Download the teacher model: ``` python download_ckpt.py ```

2. Generate AEs against the teacher model: ``` python gen_adv.py ```

### Merge all the data

There are several parameters that need to be modified before running the code, which is included in the "PARAMETER" section of the script.


```
python merge.py
```

