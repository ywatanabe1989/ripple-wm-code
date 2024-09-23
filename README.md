## Code of --- "Hippocampal neural fluctuation between memory encoding and retrieval states during a working memory task in humans"

#### [Installation and download](./docs/installation.md)

#### Data

``` bash
./data
mv ./data ./.data_bak
ln -sf ../scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix ./data/data_nix
```


#### Scripts
```bash
~/.bin/cleanup_directory.sh ./data
./scripts/load/all.sh
./scripts/demographic/all.sh
./scripts/GPFA/all.sh
./scripts/NT/all.sh

./scripts/ripple/all.sh
./scripts/memory_load/all.sh
```
