# Environmnent installation

## Conda environment creation

For successfully running this code, you need an appropriately defined conda environment.

```
conda env create -f environment.yml
```

This will create the `dl-lectures-segmentation` environment you should then be able to load with 

```
source activate dl-lectures-segmentation
```

## Original conda environment creation 

The conda environment has been created with :

```
conda create --name dl-lectures-segmentation python=3.9 --force
source activate dl-lectures-segmentation
pip install -r requirements.txt

```

# Downloading the data

The data used in this labwork are provided by Armeni et al.(2017). These can be downloaded from [http://buildingparser.stanford.edu/dataset.html#Download](http://buildingparser.stanford.edu/dataset.html#Download). Statistics on the different areas are provided by [http://buildingparser.stanford.edu/dataset.html#statistics](http://buildingparser.stanford.edu/dataset.html#statistics).

For this labwork, we only use part of the dataset and we require the following elements : 

```
/assets
	semantic_labels.json
/area_1
	/data
		/rgb
			*.png
		/semantic
			*.png
...
/area_3
	/data
		/rgb
			*.png
		/semantic
			*.png
```




# References

Armeni, Iro and Sax, Sasha and Zamir, Amir R and Savarese, Silvio, (2017). Joint 2d-3d-semantic data for indoor scene understanding, arXiv preprint arXiv:1702.01105
