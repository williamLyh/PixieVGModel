# PixieModel
### Evaluation datasets
MEN: https://aclweb.org/aclwiki/MEN_Test_Collection_(State_of_the_art) 

SimLek-999: https://fh295.github.io/simlex.html

### Generate dataset from Visual Genome
`--min_freq` to set the minimum threshold of triple occurrence.

`--pixie_dim` to set the dimension of pixie, which is 100.

`--data_pach`

`--pca-path`

```
python3 preprocessing.py \
    --VG_path=/the_path_of_your_VG_zip_data/ \
    --pixie_dim=100 \
    --data_path=/the_path_of_your_filtered_transformed_data/ \
    --pca_path=/the_path_of_your_pca_transformed_data/ \
    --pca_only=True \
    --min_freq=100
 ```

### Train the PixieModel
`--pixie_dim`

`--data_path`

`--parameter_path`

`--pca_path`

`--lr`

`--dr`

`epoch_num`

```
python3 pixie_model.py \
    --pixie_dim=100 \
    --pca_path=/the_path_of_your_pca_transformed_data/ \
    --parameter_path='parameters/' \
    --lr=0.01 \
    --dr=5e-8 \
    --epoch_num=20
```

### Evalute the PixieModel

`--pixie_dim`: Dimension of the pixie, default is 100.

`--dataset`: This could be one of the 'MEN', 'Simlek', 'RELPRON' and 'GS2011'.

`--data_path`

`--pca_path`

`--parameter_path`: The path of the trained parameters 'world_parameters.p' and 'Lexical_parameters.p'.

`--use_EVA_vocab`: Boolean value to set if only use the vocab covered by the EVA work.

```
python3 evaluation.py --dataset=Simlek --pixie_dim=100 --data_path=pixie_data_2/ --pca_path=data_pca_2/
```
