import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class YoLov11mTRT:
    def __init__(self, engine_path):
        with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_shape = (1, 3, 640, 640)
        self.output_shape = (1, 8400, 85)
        self.allocate_buffers()

    def allocate_buffers(self):
        self.input_host = np.empty(self.input_shape, dtype=np.float32)
        self.output_host = np.empty(self.output_shape, dtype=np.float32)

        self.input_device = cuda.mem_alloc(self.input_host.nbytes)
        self.output_device = cuda.mem_alloc(self.output_host.nbytes)

        self.bindings = [int(self.input_device), int(self.output_device)]
        self.stream = cuda.Stream()

    def infer(self, image):
        img = self.preprocess(image)

        # Ambil nama tensor input/output dari engine
        input_name = self.engine.get_tensor_name(0)
        output_name = self.engine.get_tensor_name(1)

        # Set input shape (jika dinamis)
        self.context.set_input_shape(input_name, img.shape)

        # Set alamat tensor
        self.context.set_tensor_address(input_name, int(self.input_device))
        self.context.set_tensor_address(output_name, int(self.output_device))

        # Copy input ke device
        cuda.memcpy_htod_async(self.input_device, img, self.stream)

        # Jalankan inference
        success = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not success:
            print("❌ Inference gagal dijalankan.")
            return [[]]

        # Ambil hasil dari device
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        return self.postprocess(self.output_host.reshape(self.output_shape))


    def preprocess(self, img):
        resized = cv2.resize(img, (640, 640))
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img.copy()

    def postprocess(self, output, conf_thresh=0.25, iou_thresh=0.45):
        predictions = output[0]  # Shape: (8400, 85)
        boxes = []

        for pred in predictions:
            obj_conf = pred[4]
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            confidence = obj_conf * class_conf

            if confidence > conf_thresh:
                cx, cy, w, h = pred[:4]
                x = cx - w / 2
                y = cy - h / 2
                boxes.append([x, y, w, h, confidence, class_id])

        if len(boxes) == 0:
            return [[]]

        boxes_xyxy = []
        confidences = []
        for box in boxes:
            x, y, w, h, conf, cls_id = box
            boxes_xyxy.append([int(x), int(y), int(x + w), int(y + h)])
            confidences.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes_xyxy, confidences, conf_thresh, iou_thresh)

        final_dets = []
        if len(indices) > 0:
            for i in indices:
                i = i[0] if isinstance(i, (list, np.ndarray)) else i
                x, y, x2, y2 = boxes_xyxy[i]
                w = x2 - x
                h = y2 - y
                conf = confidences[i]
                cls_id = int(boxes[i][5])
                final_dets.append([x, y, w, h, conf, cls_id])

        return [final_dets]
