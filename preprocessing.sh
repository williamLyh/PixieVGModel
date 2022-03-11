# wget http://visualgenome.org/static/data/dataset/objects.json.zip
python3 preprocessing.py \
    --VG_path=/local/scratch/yl535/visualgeno/ \
    --pixie_dim=100 \
    --data_path=/local/scratch/yl535/pixie_data/ \
    --pca_path=pca_data_loose/ \
    --pca_only=True \
    --min_freq=100