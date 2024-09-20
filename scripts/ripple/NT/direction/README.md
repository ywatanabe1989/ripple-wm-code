- v1, v2
    - eSWR+, rSWR+
    - eSWR+, ER
    - rSWR+, ER
    - eSWR-, rSWR-
    - eSWR-, ER
    - rSWR-, ER

| Null Hypothesis                             | Statistical Test                                           |
|---------------------------------------------|------------------------------------------------------------|
| No difference between v1 and v2 directions  | cos(v1, v2) vs. uniform distribution                       |
| No difference between SWR+ and SWR-         | cos(v1, v2) for SWR+ vs. cos(v1, v2) for SWR-              |
| No set size dependency                      | cos(v1, v2) for set sizes 4, 6, and 8                      |
|                                             | Correlation analysis with shuffled set sizes               |
| No task dependency                          | cos(v1, v2) for Match IN vs. cos(v1, v2) for Mismatch OUT  |



| Null Hypothesis                             | Statistical Test                                           | File Name (.py)           |
|---------------------------------------------|------------------------------------------------------------|---------------------------|
| No difference between v1 and v2 directions  | cos(v1, v2) vs. uniform distribution                       | vector_direction_test.py  |
| No difference between SWR+ and SWR-         | cos(v1, v2) for SWR+ vs. cos(v1, v2) for SWR-              | swr_comparison.py         |
| No set size dependency                      | cos(v1, v2) for set sizes 4, 6, and 8                      | set_size_analysis.py      |
|                                             | Correlation analysis with shuffled set sizes               | set_size_correlation.py   |
| No task dependency                          | cos(v1, v2) for Match IN vs. cos(v1, v2) for Mismatch OUT  | task_dependency_test.py   |




filenames=(vector_direction_test.py swr_comparison.py set_size_analysis.py set_size_correlation.py task_dependency_test.py )
for f in "${filenames[@]}"; do touch $f; done
