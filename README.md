# PixieModel
### Evaluation datasets
MEN: https://aclweb.org/aclwiki/MEN_Test_Collection_(State_of_the_art) 

SimLek-999: https://fh295.github.io/simlex.html

### Generate dataset from Visual Genome
```
python3 preprocessing.py --pixie_dim=100 --data_path=pixie_data_2/ --pca_path=data_pca_2/
```

### Train the PixieModel
```
python3 pixie_model.py --pixie_dim=100 --data_path=pixie_data_2/ --pca_path=data_pca_2/ --parameter_path=''
```

### Evalute the PixieModel
```
python3 evaluation.py --dataset=Simlek --pixie_dim=100 --data_path=pixie_data_2/ --pca_path=data_pca_2/
```
