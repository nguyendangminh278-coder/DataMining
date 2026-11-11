# Data sources (not uploaded to public GitHub)

Download Rossmann Store Sales data from [Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales/data) or a team-owned Drive copy after checking redistribution permissions. Place the original CSVs inside `data/raw/`:

- `train.csv`
- `test.csv`
- `store.csv`
- `sample_submission.csv`

Run `python tools/prepare_data_layout.py` to create local aliases. Additional cleaned or intermediate P09/P14 CSV files are separate project outputs, not part of this GitHub repository. See [source manifest](../docs/source_manifest.csv) for link locations.
