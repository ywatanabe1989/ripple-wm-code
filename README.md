## Code of --- "Hippocampal neural fluctuation between memory encoding and retrieval states during a working memory task in humans"

#### [Installation and download](./docs/installation.md)

#### Data

``` bash
./data
```


#### Scripts
```bash
./scripts/load/all.sh
./scripts/demographic/all.sh
./scripts/ripple/all.sh
./scripts/GPFA/all.sh


## NT
./scripts/NT/znorm_NT.py

./scripts/NT/visualization/kde_8_factors.py
./scripts/NT/visualization/scatter_kde.py # good for set-size separations
./scripts/NT/visualization/umap_8_factors.py # fxime; colors, supervised

# Classification
./scripts/NT/clf/linearSVC.py

# Distance
./scripts/NT/distance/from_O/MTL_regions.py
./scripts/NT/distance/between_gs/geometrics_medians.py
./scripts/NT/distance/between_gs/calc_dists.py
./scripts/NT/distance/between_gs/dists_stats.py
./scripts/NT/distance/between_gs/MTL_regions.py

# Memory-load Dependancies
./scripts/memory-load/all.sh
./scripts/memory-load/distance_between_gs.py
```

#### Figures

``` bash
./scripts/figures/01.py
./scripts/figures/02.py
./scripts/figures/03.py
```















#### Wavelet transformation

``` bash
./scripts/calc_wavelet.py
```









./EDA/check_ripples/
./EDA/check_ripples/unit_firing_patterns/
./EDA/check_ripples/unit_firing_patterns/trajectory/


#### Distance between phases
```
./EDA/check_ripples/unit_firing_patterns/trajectory/peri_SWR_dist_from_P_dev.py
```

#### Representative Trajectory of Subject 06, Session 02
```
./EDA/check_ripples/unit_firing_patterns/trajectory/repr_traj.py
./res/figs/scatter/repr_traj/session_traj_Subject_06_Session_02.csv
```

#### Representative Trajectory of Subject 06, Session 02 by condition
```
./EDA/check_ripples/unit_firing_patterns/trajectory/repr_traj_by_set_size_and_task_type.py 
./res/figs/scatter/repr_traj/

./EDA/check_ripples/unit_firing_patterns/trajectory/classify_trajectories.py 


```


