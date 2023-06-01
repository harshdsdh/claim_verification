mkdir -p data
mkdir -p models
curl -o data/train.jsonl https://fever.ai/download/feverous/feverous_train_challenges.jsonl
curl -o data/dev.jsonl https://fever.ai/download/feverous/feverous_dev_challenges.jsonl
curl -o data/test_unlabeled.jsonl https://fever.ai/download/feverous/feverous_test_unlabeled.jsonl
curl -o data/feverous-wiki-pages-db.zip https://fever.ai/download/feverous/feverous-wiki-pages-db.zip
unzip data/feverous-wiki-pages-db.zip