# Benchmarking YOLOv5

## Benchmark DeepSparse

Run the following for usage:

```bash
python3 benchmark_deepsparse.py --help
```

Runs at batch=1:
```bash
python3 benchmark_deepsparse.py --pipeline --engine
```

Runs at batch=64:
```bash
python3 benchmark_deepsparse.py --pipeline --engine --batch_size 64 --iterations 5
```

## Benchmark GPU
