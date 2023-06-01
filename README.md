[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)



1. steps to install dependencies: 
    - `pip install -r requirements.txt`

2. steps to download data: 
    - `bash download_data.sh`

3. steps to process data: 
    - `python process_data.py --train_path ../data/train.jsonl --dev_path ../data/dev.jsonl`

4. steps to train the model from root of repo: 
    - `python src/train.py`
