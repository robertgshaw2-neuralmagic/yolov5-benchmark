import os

import time, cv2, argparse
from deepsparse import Engine, Pipeline
from deepsparse.yolo.utils import postprocess_nms
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default="zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none")
parser.add_argument('--image_path', type=str, default="basilica.jpg")
parser.add_argument('--do_pipeline', action='store_true')
parser.add_argument('--do_engine', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--iterations', type=int, default=100)

def benchmark_engine(compiled_model, image, batch_size=1, iterations=100):
    data = np.ascontiguousarray(np.stack([np.moveaxis(image,-1,0)]*batch_size))
   
    start = time.perf_counter()
    for _ in range(iterations):
        _ = compiled_model([data])
    end = time.perf_counter()
    
    throughput = (iterations*batch_size) / (end-start)
    print(f"Engine Throughput: {round(throughput,2)}")
    return throughput

def benchmark_pipeline(pipeline, image, batch_size=1, iterations=100):
    data = [image]*batch_size
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = pipeline(images=data)
    end = time.perf_counter()

    throughput = (iterations*batch_size) / (end-start)
    print(f"Pipeline Throughput: {round(throughput,2)}")
    
    return throughput

if __name__ == '__main__':
    args = parser.parse_args()
    
    im = Image.open(args.image_path)
    np_im = np.array(im)
    np_im_640 = np.array(im.resize((640,640)))
    
    print("Compiling...")
    compiled_model = Engine(model=args.model_path, batch_size=args.batch_size)
    compiled_pipeline = Pipeline.create(task="yolo", model_path=args.model_path, batch_size=args.batch_size)
    
    if args.do_engine:
        print("\nBenchmarking Engine...")
        _ = benchmark_engine(compiled_model, np_im_640, args.batch_size, args.iterations)
    
    if args.do_pipeline:
        print("\nBenchmarking Pipeline...")
        _ = benchmark_pipeline(compiled_pipeline, np_im_640, args.batch_size, args.iterations)    
        
    
