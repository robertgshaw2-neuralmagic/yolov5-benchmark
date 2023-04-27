import time, argparse, torch
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default="yolov5s")
parser.add_argument('--image_path', type=str, default="basilica.jpg")
parser.add_argument('--big_image', action='store_true')
parser.add_argument('--do_model', action='store_true')
parser.add_argument('--do_pipeline', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--do_cpu', action='store_true')

def benchmark_model(model, image, batch_size=1, iterations=100):
    batch = torch.tensor(np.stack([im]*batch_size, axis=0))
        
    start = time.perf_counter()
    for _ in range(iterations):
        _ = model(batch)
    end = time.perf_counter()
    
    throughput = (iterations*batch_size) / (end-start)
    print(f"Engine Throughput: {round(throughput,2)}")
    return throughput

if __name__ == '__main__':
    args = parser.parse_args()
    
    if not args.do_cpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    im = Image.open(args.image_path)    
    if not args.big_image:   
        im = im.resize((640,640))
    im = np.array(im).astype(np.float32) / 255.
    im = np.moveaxis(im, -1, 0)
    
    print("Downloading...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.to(device)
    
    if args.do_model:
        print("\nBenchmarking Model...")
        print(f"Running with torch.__version__=={torch.__version__} on device=={device}") 
        _ = benchmark_model(model=model, image=im, batch_size=args.batch_size, iterations=args.iterations)
    
    if args.do_pipeline:
        print("\nBenchmarking Pipeline...")
        print("Not implemented yet")
        
    
