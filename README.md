# Smart Motion Sensor

Motion detection and signal generation inspired by neuromorphic computing. Instead of processing every video frame in full, this project focuses computation on regions where meaningful change occurs, detecting relevant moving objects (cars, motorcycles, bicycles, people) from fixed-camera footage.

## Pipeline

```
Image -> Change Detection -> Changed Region -> Object Size/Movement Analysis -> Relevant Object -> Signal
```

The project is built in six stages:

1. **Dataset and Matrix Preparation** - load the fixed-camera grayscale dataset, verify resolution and pixel format, represent each frame as a pixel matrix.
2. **Image/Region Analysis** - group frames by spatial and visual similarity.
3. **Pixel Difference and Motion Analysis** - detect boundary pixels between consecutive frames, filter them by pixel variance, and compute displacement (dx, dy).
4. **Threshold Determination** - tune pixel-difference and variance thresholds so noise is filtered out while real motion is kept.
5. **Motion Detection and Signal Generation** - fire a signal when a relevant object's movement crosses the determined thresholds.
6. **Neuromorphic Extension** - represent significant changes as events/spikes and explore a spiking neural network (SNN) based approach.

## Status

| Stage | Status |
|---|---|
| 1. Dataset and Matrix Preparation | Done |
| 2. Image/Region Analysis | In progress |
| 3. Pixel Difference and Motion Analysis | In progress |
| 4. Threshold Determination | Not started |
| 5. Motion Detection and Signal Generation | Not started |
| 6. Neuromorphic Extension | Not started |

## Datasets

Two fixed-camera grayscale sequences, provided as part of the project:

- `data/car-turn` - 80 frames, a car completing a turn
- `data/motorbike` - 43 frames, a motorbike in motion

Both are 854x480, single-channel grayscale.

## Project structure

```
Smart-Motion-Sensor/
├── data/           sequences go here (not tracked in git)
├── outputs/        generated plots and CSVs
├── src/
│   └── dataset_prep.py       Stage 1
└── README.md
```

More scripts will be added to `src/` as later stages are completed.

## Running it

```
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python matplotlib scipy

python3 src/dataset_prep.py
```

Run scripts from the project root, not from inside `src/`, since file paths are relative to the root.

## References

- Gallego et al., "Event-based Vision: A Survey," IEEE TPAMI, 2020.
- Atiq et al., "Vehicle detection and shape recognition using optical sensors: a review," ICMLC, 2010.
