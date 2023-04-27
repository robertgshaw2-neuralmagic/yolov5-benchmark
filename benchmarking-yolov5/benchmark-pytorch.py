import time, cv2, argparse, torch
from deepsparse.yolo.utils import postprocess_nms
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default="zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/pruned85_quant-none")
parser.add_argument('--image_path', type=str, default="basilica.jpg")
parser.add_argument('--big_image', action='store_true')
parser.add_argument('--do_pipeline', action='store_true')
parser.add_argument('--do_engine', action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--do_cpu', action='store_true')

def benchmark_engine(compiled_model, image, batch_size=1, iterations=100):
    data = np.ascontiguousarray(np.stack([np.moveaxis(image,-1,0)]*batch_size))
   
    start = time.perf_counter()
    for _ in range(iterations):
        _ = compiled_model([data])
    end = time.perf_counter()
    
    throughput = (iterations*batch_size) / (end-start)
    print(f"Engine Throughput: {round(throughput,2)}")
    return throughput

def benchmark_pipeline(compiled_model, image, batch_size=1, iterations=100):
    original_shape = None
    if image.shape[:2] != (640,640):
        original_shape = image.shape[:2]
    batch = [image]*batch_size
    
    executor = ThreadPoolExecutor(max_workers=8)
    
    start = time.perf_counter()
    for _ in range(iterations):
        engine_inputs = [preprocess(batch, executor)]
        engine_outputs = compiled_model(engine_inputs)
        outputs = postprocess(engine_outputs, original_image_shape=original_shape)
    end = time.perf_counter()

    throughput = (iterations*batch_size) / (end-start)
    print(f"Pipeline Throughput: {round(throughput,2)}")
    
    return throughput

def preprocess_img(input_img, target_shape=(640,640)):
    if input_img.shape[:2] != target_shape:
        im = cv2.resize(input_img, dsize=target_shape)
    else:
        im = input_img
    im = np.moveaxis(im,-1,0)
    return im

def preprocess(inputs, executor):
    image_batch = list(executor.map(preprocess_img, inputs))
    return torch.tensor(
        np.stack(image_batch,axis=0), 
        dtype=torch.float32, 
        device=device) / 255.0

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

if __name__ == '__main__':
    args = parser.parse_args()
    
    if not args.do_cpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    im = Image.open(args.image_path)
    np_im = np.array(im)
    np_im_640 = np.array(im.resize((640,640)))
    
    print("Compiling...")
    compiled_model = Engine(model=args.model_path, batch_size=args.batch_size)
    
    if args.do_engine:
        print("\nBenchmarking Engine...")
        _ = benchmark_engine(compiled_model, np_im_640, args.batch_size, args.iterations)
    
    if args.do_pipeline:
        print("\nBenchmarking Pipeline...")
  
        if not args.big_image:
            _ = benchmark_pipeline(compiled_model, np_im, args.batch_size, args.iterations)
        else:
            _ = benchmark_pipeline(compiled_model, np_im_640, args.batch_size, args.iterations)    
        
    