[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeremyfix/deeplearning-lectures/blob/lightning/LabsSolutions/00-pytorch-FashionMNISTLightning/illustration.ipynb)

# Installation

To install the requirements, simply

```
virtualenv -p python3 venv
source venc/bin/activate
python -m pip install -r requirements.txt
```

And then you are ready to go

# Running experiments

To run an experiment :

	python train.py --model fancyCNN --data_augment

To run experiments with NeptuneAI :

	export NEPTUNE_TOKEN=YOUR_NEPTUNE_TOKEN
	export NEPTUNE_PROJECT=YOUR_NEPTUNE_PROJECT
	python train.py --model fancyCNN --data_augment


