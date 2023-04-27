# Benchmarking YOLOv5

Download image:
```bash
wget -O basilica.jpg https://raw.githubusercontent.com/neuralmagic/deepsparse/main/src/deepsparse/yolo/sample_images/basilica.jpg
```

## Benchmark DeepSparse

Run the following for usage:

```bash
python3 benchmark-deepsparse.py --help
```

Runs at batch=1:
```bash
python3 benchmark-deepsparse.py --pipeline --engine
```

Runs at batch=64:
```bash
python3 benchmark-deepsparse.py --pipeline --engine --batch_size 64 --iterations 5
```

## Benchmark GPU
