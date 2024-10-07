## Code of --- "Hippocampal neural fluctuation between memory encoding and retrieval states during a working memory task in humans"

#### [Installation and download](./docs/installation.md)

#### Data

``` bash
./data
mv ./data ./.data_bak
ln -sf ../scripts/externals/Human_MTL_units_scalp_EEG_and_iEEG_verbal_WM/data_nix ./data/data_nix
```

#### Cuda

``` bash
~/.bin/nvidia-install-nvidia-driver
~/.bin/nvidia-install-cuda
~/.bin/nvidia-check-cuda 2> /dev/null | grep -i "is cuda available" # True
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


#### Project Structure

``` bash
tree . --gitignore > ./docs/tree-project.txt
tree ./data -l > ./docs/tree-data.txt
```


#### Scripts NT

``` bash
./scripts/NT/znorm_NT.py
./scripts/NT/calc_geometrics_medians.py

# ./scripts/NT/visualization/kde_8_factors.py
# ./scripts/NT/visualization/scatter_kde.py # suitable for set-size separations
# ./scripts/NT/visualization/umap_8_factors.py # fixme; colors, supervised

# Distance from O
./scripts/NT/distance/from_O/MTL_regions.py

# Distance between geometrics medians
./scripts/NT/distance/between_gs/MTL_regions.py

./scripts/NT/distance/between_gs/calc_dist/trial.py
./scripts/NT/distance/between_gs/calc_dist/match_set_size.py
./scripts/NT/distance/between_gs/calc_dist/session.py

# Rank
./scripts/NT/distance/between_gs/rank_dists/to_rank_dists.py
./scripts/NT/distance/between_gs/rank_dists/stats.py

# Set size dependency
./scripts/NT/distance/between_gs/set_size_dependency/plot_box.py
# ./scripts/NT/distance/between_gs/set_size_dependency/stats.py
# ./scripts/NT/distance/between_gs/set_size_dependency/plot_violin.py

# # Classification
# ./scripts/NT/clf/linearSVC.py
```

