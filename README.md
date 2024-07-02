## Code of --- "Hippocampal neural fluctuation between memory encoding and retrieval states during a working memory task in humans"

#### [Installation and download](./docs/installation.md)

#### Converts the .h5 files into csv and pkl files
```bash
./scripts/load/nix_2_csv_and_pkl.py
```

#### Demographic data
```bash
./scripts/demographic/fetch_demographic_data.py
```

#### Ripple Detection
```bash
./scripts/ripple/detect_SWR_p.py
./scripts/ripple/define_SWR_m.py
# ./scripts/ripple/UMAP_for_defining_putative_CA1.py
```

#### Neural trajectory (NT) calculation with GPFA
```bash
find data -name '*NT*' | xargs rm -rf
./scripts/NT/calc_NT_with_GPFA.py
./scripts/NT/znorm_NT.py
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


