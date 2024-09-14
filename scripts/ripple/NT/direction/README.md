To calculate the speed of a trajectory in 3D space:

1. Calculate the displacement between consecutive points:
   Δx = x2 - x1
   Δy = y2 - y1
   Δz = z2 - z1

2. Calculate the distance using the Euclidean formula:
   distance = sqrt(Δx^2 + Δy^2 + Δz^2)

3. Divide the distance by the time interval between points:
   speed = distance / Δt

Here's a Python implementation:

```python
import numpy as np

def calculate_speed(trajectory, timestamps):
    displacements = np.diff(trajectory, axis=0)
    distances = np.linalg.norm(displacements, axis=1)
    time_intervals = np.diff(timestamps)
    speeds = distances / time_intervals
    return speeds

```
