# Hand Gesture Recognition

Two complementary approaches:

| Approach | File | Needs training | Best for |
|---|---|---|---|
| CNN on images | `train.py`, `realtime_cnn.py` | Yes (LeapGestRecog) | Image/video classification, report metrics |
| MediaPipe landmarks | `realtime_mediapipe.py` | No | Live gesture-based control |

## Setup
```
pip install -r requirements.txt
```
Download the dataset from https://www.kaggle.com/datasets/gti-upm/leapgestrecog and unzip.

## Train
```
python train.py --data path/to/leapGestRecog --epochs 15
```
Outputs: `gesture_model.keras`, `classes.json`, `confusion_matrix.png`, `training_curves.png`.
Split is by subject (00-06 train / 07-08 val / 09 test) to measure generalisation to new people.

## Run live
```
python realtime_cnn.py          # CNN
python realtime_mediapipe.py    # landmark-based, triggers actions
```
