# ISE-YOLO

You can test the weights to get the results in the paper with the following command

```bash
python val_map.py --weights ./weights/ISE-YOLO-L.pt --data data/infraredImg.yaml --batch-size 1 --task test --half --save-json
python val_map.py --weights ./weights/ISE-YOLO-S.pt --data data/infraredImg.yaml --batch-size 1 --task test --half --save-json
```

| Method     | $AP$ | $AP_{50}$ | $AP_{75}$ | $AP_{S}$ | $AP_{M}$ | $AP_{L}$ | $AR$ | Params |
| ---------- | ---- | --------- | --------- | -------- | -------- | -------- | ---- | ------ |
| ISE-YOLO-L | 51.9 | 80.6      | 58.7      | 27.0     | 36.8     | 57.7     | 62.2 | 27.3   |
| ISE-YOLO-S | 49.0 | 78.9      | 53.4      | 17.9     | 35.2     | 54.8     | 61.6 | 5.5    |

Only the test code, the test dataset, and some of the training and validation sets are currently publicly available. The full training code and dataset will be uploaded when the paper is accepted.