import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

cuda.init()  # Inisialisasi manual (bisa dihapus karena sudah ada autoinit)

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

    def infer(self, image):  # image: (H, W, 3) in BGR
        img = self.preprocess(image)
        cuda.memcpy_htod_async(self.input_device, img, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=int(self.stream.handle))
        cuda.memcpy_dtoh_async(self.output_host, self.output_device, self.stream)
        self.stream.synchronize()

        return self.postprocess(self.output_host.reshape(self.output_shape))

    def preprocess(self, img):
        resized = cv2.resize(img, (640, 640))
        img = resized.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img

    def postprocess(self, output):
        return output  # Tambahkan NMS/visualisasi di luar
