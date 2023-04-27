import time, argparse, torch, torchvision
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms
from typing import Union
from helpers import postprocess_nms

transform = transforms.Compose([
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

parser = argparse.ArgumentParser()

parser.add_argument('--image_path', type=str, default="basilica.jpg")
parser.add_argument('--do_model', action='store_true')
parser.add_argument('--do_pipeline', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--do_cpu', action='store_true')

def benchmark_model(model, image, batch_size=1, iterations=100):
    batch = torch.tensor(np.stack([image]*batch_size, axis=0))
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(batch)
        end = time.perf_counter()
    
    throughput = (iterations*batch_size) / (end-start)
    print(f"Engine Throughput: {round(throughput,2)}")
    return throughput

def preprocess_img(input_img):
    return transform(input_img)

def preprocess(inputs, executor):
    image_batch = list(executor.map(preprocess_img, inputs))
    return torch.stack(image_batch)

def scale_boxes(boxes, original_image_shape):
    if not original_image_shape:
        return boxes

    scale = np.flipud(
        np.divide(
            np.asarray(original_image_shape), np.asarray((640,640))
        )
    )
    
    scale = np.concatenate([scale, scale])
    boxes = np.multiply(boxes, scale)
    return boxes

def postprocess(engine_outputs, original_image_shape=None):
    outputs = postprocess_nms(
        outputs=engine_outputs,
        iou_thres=0.25,
        conf_thres=0.45,
        multi_label=False
    )

    batch_boxes, batch_scores, batch_labels = [], [], []
    
    for idx, image_output in enumerate(outputs):
        batch_boxes.append(
            scale_boxes(
                boxes=image_output[:, 0:4],
                original_image_shape=original_image_shape,
            ).tolist(),
        )
        batch_scores.append(image_output[:, 4].tolist())
        batch_labels.append(image_output[:, 5].tolist())

    return batch_boxes, batch_scores, batch_labels

def benchmark_pipeline(model, image, batch_size=1, iterations=100):
    executor = ThreadPoolExecutor(max_workers=8)
    batch = [image]*batch_size
    
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(iterations):
            model_inputs = preprocess(batch, executor)
            model_outputs = model(model_inputs)
            outputs = postprocess(model_outputs) 
        end = time.perf_counter()
    
    throughput = (iterations*batch_size) / (end-start)
    print(f"Pipeline Throughput: {round(throughput,2)}")
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    if not args.do_cpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    im = Image.open(args.image_path)
    im = im.resize((640,640))
    im_np = np.array(im).astype(np.float32) / 255.
    im_np = np.moveaxis(im_np, -1, 0)

    print("Downloading...")
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    model.to(device)
    
    if args.do_model:
        print("\nBenchmarking Model...")
        print(f"Running with torch.__version__=={torch.__version__} on device=={device}") 
        _ = benchmark_model(model=model, image=im_np, batch_size=args.batch_size, iterations=args.iterations)
    
    if args.do_pipeline:
        print("\nBenchmarking Pipeline...")
        print(f"Running with torch.__version__=={torch.__version__} on device=={device}") 
        _ = benchmark_pipeline(model=model, image=im, batch_size=args.batch_size, iterations=args.iterations)
        
    
