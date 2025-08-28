import cv2
import numpy as np
import time
import threading
import json
import requests
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from datetime import datetime
import yaml

# TensorRT / PyCUDA imports (required on Jetson Nano runtime)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
except ImportError:
    trt = None
    cuda = None
    print("Warning: TensorRT or PyCUDA not found. This script will not run.")


class TRTDetector:
    """
    A class for performing object detection using a TensorRT engine.
    Handles model loading, preprocessing, inference, and post-processing.
    """
    def __init__(self, engine_path, conf_threshold=0.25, iou_threshold=0.45):
        if trt is None or cuda is None:
            raise RuntimeError("TensorRT/PyCUDA not available. Please install on your Jetson device.")

        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"TensorRT engine not found at: {engine_path}")

        self.engine_path = engine_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)

        # Initialize TensorRT logger, runtime, and engine
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize the TensorRT engine.")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create a TensorRT execution context.")

        # Allocate memory for inputs and outputs
        self._allocate_buffers()

        # Create a CUDA stream for asynchronous execution
        self.stream = cuda.Stream()

    def _allocate_buffers(self):
        """
        Allocates host and device memory for engine bindings (inputs/outputs).
        """
        self.bindings = []
        self.host_inputs = []
        self.host_outputs = []
        self.device_inputs = []
        self.device_outputs = []

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # Append the device buffer address to the bindings list
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.input_shape = self.engine.get_binding_shape(binding)
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def _letterbox(self, image, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Resizes and pads an image to a new shape while maintaining the aspect ratio.
        This is a standard preprocessing step for YOLO models.
        """
        h, w = image.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        
        new_unpad = (int(round(w * r)), int(round(h * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if (w, h) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return image, r, (left, top)

    def _postprocess(self, outputs, input_hw, orig_hw, pad, scale):
        """
        Parses the raw model output to generate final bounding boxes.
        This is the corrected and simplified post-processing logic.
        """
        H_in, W_in = input_hw
        H0, W0 = orig_hw
        pad_left, pad_top = pad  # Fix: letterbox returns (left, top)

        # Handle different output formats
        if len(outputs) == 0:
            return []

        output = outputs[0]

        # Check output shape and format
        if output.ndim == 3:
            # Format: (1, num_proposals, num_classes + 4)
            output = output.squeeze(0)  # Remove batch dimension
        elif output.ndim == 2:
            # Format: (num_proposals, num_classes + 4) - already correct
            pass
        else:
            # Log warning for unexpected shapes
            print(f"Warning: Unexpected output shape: {output.shape}, expected 2D or 3D")
            return []

        if output.shape[0] == 0:
            return []

        if output.shape[1] < 5:
            print(f"Warning: Output has {output.shape[1]} columns, expected at least 5 (cx,cy,w,h,conf)")
            return []

        # In YOLO format: [cx, cy, w, h, obj_conf, class_prob_1, class_prob_2, ...]
        boxes_coords = output[:, :4]  # [cx, cy, w, h]
        obj_conf = output[:, 4]  # Object confidence

        if output.shape[1] > 5:
            # Multiple classes: take max class probability
            class_confs = output[:, 5:]
            class_ids = np.argmax(class_confs, axis=1)
            class_conf_max = np.max(class_confs, axis=1)
            confidences = obj_conf * class_conf_max
        else:
            # Single class or binary classification
            class_ids = np.zeros(len(output), dtype=int)
            confidences = obj_conf

        # Filter out detections below the confidence threshold
        mask = confidences > self.conf_threshold
        boxes_coords = boxes_coords[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_coords) == 0:
            return []

        # Check if coordinates are normalized (typically 0-1 range)
        # Be more conservative with the threshold to avoid false positives
        if np.max(boxes_coords) <= 1.1:
            # Scale normalized coordinates to letterboxed image space
            boxes_coords[:, [0, 2]] *= W_in  # cx, w
            boxes_coords[:, [1, 3]] *= H_in  # cy, h

        # Convert [center_x, center_y, width, height] to [x1, y1, x2, y2]
        cx, cy, w, h = boxes_coords[:, 0], boxes_coords[:, 1], boxes_coords[:, 2], boxes_coords[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Prepare boxes for NMS (cv2.dnn.NMSBoxes expects [x, y, w, h])
        nms_boxes = []
        for i in range(len(x1)):
            box_x1, box_y1, box_x2, box_y2 = x1[i], y1[i], x2[i], y2[i]
            box_w = box_x2 - box_x1
            box_h = box_y2 - box_y1
            nms_boxes.append([int(box_x1), int(box_y1), int(box_w), int(box_h)])

        # Perform Non-Maximum Suppression (NMS)
        try:
            indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), self.conf_threshold, self.iou_threshold)
            if len(indices) == 0:
                return []
            # Handle different OpenCV versions
            if isinstance(indices, np.ndarray) and indices.ndim > 1:
                indices = indices.flatten()
        except Exception as e:
            self.logger.log(trt.Logger.WARNING, f"NMS failed: {e}")
            return []

        detections = []
        for i in indices:
            if isinstance(i, (list, np.ndarray)):
                i = i[0] if len(i) > 0 else 0

            # Get the box coordinates in the letterboxed image space
            box_x, box_y, box_w, box_h = nms_boxes[i]

            # Map coordinates back from letterboxed space to original image space
            final_x1 = (box_x - pad_left) / scale
            final_y1 = (box_y - pad_top) / scale
            final_x2 = (box_x + box_w - pad_left) / scale
            final_y2 = (box_y + box_h - pad_top) / scale

            # Ensure minimum box size and clip coordinates
            final_x1 = max(0, min(final_x1, W0 - 1))
            final_y1 = max(0, min(final_y1, H0 - 1))
            final_x2 = max(0, min(final_x2, W0 - 1))
            final_y2 = max(0, min(final_y2, H0 - 1))

            # Skip very small boxes
            if (final_x2 - final_x1) < 2 or (final_y2 - final_y1) < 2:
                continue

            detections.append([
                float(final_x1), float(final_y1),
                float(final_x2), float(final_y2),
                float(confidences[i]), int(class_ids[i])
            ])

        return detections

    def infer(self, image_bgr):
        """
        Performs a full inference cycle: preprocess, inference, and post-process.
        """
        # --- Preprocessing ---
        _, c, H, W = self.input_shape
        assert c == 3, "Model expects a 3-channel input image"
        
        img_letter, scale, pad = self._letterbox(image_bgr, (H, W))
        img_rgb = cv2.cvtColor(img_letter, cv2.COLOR_BGR2RGB)
        
        # Transpose from HWC to CHW and normalize to [0, 1]
        input_data = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
        input_data = np.ascontiguousarray(input_data)
        
        # Copy preprocessed image data to the host buffer
        np.copyto(self.host_inputs[0], input_data.ravel())

        # --- Inference ---
        # Transfer input data from host to device (GPU)
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        
        # Execute the model
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        # Transfer output data from device back to host
        for i in range(len(self.host_outputs)):
            cuda.memcpy_dtoh_async(self.host_outputs[i], self.device_outputs[i], self.stream)
            
        # Wait for the stream to finish all operations
        self.stream.synchronize()

        # --- Post-processing ---
        # Reshape the flattened output to its original shape
        outputs = [out.reshape(self.engine.get_binding_shape(i+1)) for i, out in enumerate(self.host_outputs)]
        
        H0, W0 = image_bgr.shape[:2]
        detections = self._postprocess(outputs, (W, H), (W0, H0), pad, scale)
        
        return detections


class GlaucomaApplicationTRT(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Glaucoma Detection System (TensorRT)")
        self.geometry("1200x900")
        self.resizable(False, False)

        # Initialize variables
        self.patient_info = {}
        self.doctor_info = {}
        self.detection_result = None
        self.result_image = None
        self.original_image_path = None
        self.server_upload_url = os.environ.get("GL_UPLOAD_URL", "http://100.104.212.73:8002/uploads")
        self.server_record_url = os.environ.get("GL_RECORD_URL", "http://100.104.212.73:8002/record/create")
        self.server_cookie = os.environ.get("GL_SERVER_COOKIE", "")
        
        # Mode control: 'camera' or 'image'
        self.detection_mode = 'camera'
        self.imported_image = None
        self.imported_image_path = None
        self.display_locked = False

        # Backend info
        self.backend = 'tensorrt'
        self.device = 'gpu'
        
        self.create_widgets()
        self.setup_model()
        
        self.cap = None
        self.frame = None
        self.ret = False
        self.boxes = []
        self.confs = []
        self.clss = []
        self.total_fps = []
        self.processing_started = False
        self.camera_running = True

        self.switch_mode()
        
        self.update_frames_thread = threading.Thread(target=self.update_frames)
        self.update_frames_thread.daemon = True
        self.update_frames_thread.start()

    def setup_model(self):
        """Initialize the TensorRT engine for glaucoma detection."""
        self.model_path = os.path.join("model_results", "runs", "train", "glaucoma_yolo11", "weights", "best.engine")
        
        try:
            conf_th = float(os.environ.get("GL_TRT_CONF", "0.25"))
            iou_th = float(os.environ.get("GL_TRT_IOU", "0.45"))
            
            self.detector = TRTDetector(self.model_path, conf_threshold=conf_th, iou_threshold=iou_th)
            
            self.class_names = self._resolve_class_names()
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
            
            self.log_to_console(f"TensorRT engine loaded successfully: {self.model_path}")
            self.log_to_console(f"Classes: {list(self.class_names.values())}")
        except Exception as e:
            self.log_to_console(f"FATAL: Engine loading failed: {e}")
            messagebox.showerror("Error", f"Failed to load TensorRT engine: {e}")

    def _resolve_class_names(self):
        """Loads class names from a YAML file or uses a default."""
        yaml_path = os.path.join("model_results", "data", "glaucoma.yaml")
        try:
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    names_list = data.get('names')
                    if isinstance(names_list, list) and len(names_list) > 0:
                        self.log_to_console(f"Loaded class names from YAML: {yaml_path}")
                        return {i: n for i, n in enumerate(names_list)}
        except Exception as e:
            self.log_to_console(f"Warning: Could not read class names from YAML: {e}")

        self.log_to_console("Warning: Falling back to default class names: ['healthy', 'glaucoma']")
        return {0: 'healthy', 1: 'glaucoma'}

    # --- GUI Creation Methods (unchanged from original) ---
    def create_widgets(self):
        self.create_header()
        self.create_main_content()
        self.create_console()

    def create_header(self):
        header_frame = tk.Frame(self)
        header_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        title_label = tk.Label(header_frame, text="Glaucoma Detection System (TensorRT)", 
                              font=("Helvetica", 24, "bold"))
        title_label.pack(pady=10)

    def create_main_content(self):
        self.create_patient_form()
        self.create_camera_section()
        self.create_doctor_results_section()

    def create_patient_form(self):
        patient_frame = tk.LabelFrame(self, text="Patient Information", font=("Helvetica", 12, "bold"))
        patient_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew", ipadx=10, ipady=10)
        tk.Label(patient_frame, text="Patient ID:").grid(row=0, column=0, sticky="w", pady=5)
        self.patient_id_var = tk.StringVar()
        tk.Entry(patient_frame, textvariable=self.patient_id_var, width=20).grid(row=0, column=1, pady=5)
        tk.Label(patient_frame, text="Patient Name:").grid(row=1, column=0, sticky="w", pady=5)
        self.patient_name_var = tk.StringVar()
        tk.Entry(patient_frame, textvariable=self.patient_name_var, width=20).grid(row=1, column=1, pady=5)
        tk.Label(patient_frame, text="Age:").grid(row=2, column=0, sticky="w", pady=5)
        self.patient_age_var = tk.StringVar()
        tk.Entry(patient_frame, textvariable=self.patient_age_var, width=20).grid(row=2, column=1, pady=5)
        tk.Label(patient_frame, text="Gender:").grid(row=3, column=0, sticky="w", pady=5)
        self.patient_gender_var = tk.StringVar()
        gender_combo = ttk.Combobox(patient_frame, textvariable=self.patient_gender_var, 
                                   values=["Male", "Female", "Other"], width=17, state="readonly")
        gender_combo.grid(row=3, column=1, pady=5)
        tk.Label(patient_frame, text="Phone:").grid(row=4, column=0, sticky="w", pady=5)
        self.patient_phone_var = tk.StringVar()
        tk.Entry(patient_frame, textvariable=self.patient_phone_var, width=20).grid(row=4, column=1, pady=5)
        tk.Label(patient_frame, text="Medical History:").grid(row=5, column=0, sticky="w", pady=5)
        self.patient_history_text = tk.Text(patient_frame, width=20, height=4)
        self.patient_history_text.grid(row=5, column=1, pady=5)
        tk.Button(patient_frame, text="Clear Form", command=self.clear_patient_form,
                 bg="#ffcccc").grid(row=6, column=0, columnspan=2, pady=10)

    def create_camera_section(self):
        camera_frame = tk.LabelFrame(self, text="Eye Examination", font=("Helvetica", 12, "bold"))
        camera_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        mode_frame = tk.Frame(camera_frame)
        mode_frame.grid(row=0, column=0, pady=5)
        tk.Label(mode_frame, text="Detection Mode:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=5)
        self.mode_var = tk.StringVar(value="camera")
        tk.Radiobutton(mode_frame, text="Camera Stream", variable=self.mode_var, value="camera",
                      command=self.switch_mode).grid(row=0, column=1, padx=5)
        tk.Radiobutton(mode_frame, text="Import Image", variable=self.mode_var, value="image",
                      command=self.switch_mode).grid(row=0, column=2, padx=5)
        self.camera_label = tk.Label(camera_frame, text="Camera Stream")
        self.camera_label.grid(row=1, column=0, pady=10)
        self.camera_frame_widget = tk.Label(camera_frame)
        self.camera_frame_widget.grid(row=2, column=0, pady=10)
        control_frame = tk.Frame(camera_frame)
        control_frame.grid(row=3, column=0, pady=10)
        self.camera_controls_frame = tk.Frame(control_frame)
        self.camera_controls_frame.grid(row=0, column=0, columnspan=3, pady=5)
        tk.Label(self.camera_controls_frame, text="Select Camera:").grid(row=0, column=0, padx=5, pady=5)
        self.camera_var = tk.StringVar(value="None")
        self.camera_dropdown = ttk.Combobox(self.camera_controls_frame, textvariable=self.camera_var, width=15)
        self.camera_dropdown.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(self.camera_controls_frame, text="Refresh Cameras", command=self.refresh_cameras
                 ).grid(row=0, column=2, padx=5, pady=5)
        tk.Button(self.camera_controls_frame, text="Apply Camera", command=self.apply_selected_camera
                 ).grid(row=1, column=0, padx=5, pady=5)
        self.image_controls_frame = tk.Frame(control_frame)
        self.image_controls_frame.grid(row=1, column=0, columnspan=3, pady=5)
        tk.Button(self.image_controls_frame, text="Import Image", command=self.import_image,
                 bg="#ffcc99").grid(row=0, column=0, padx=5, pady=5)
        self.image_path_label = tk.Label(self.image_controls_frame, text="No image selected", 
                                        wraplength=300, justify="left")
        self.image_path_label.grid(row=0, column=1, padx=5, pady=5)
        self.start_button = tk.Button(control_frame, text="Start Live Detection", 
                                     command=self.toggle_detection, bg="#ccffcc")
        self.start_button.grid(row=2, column=0, padx=5, pady=5)
        self.capture_button = tk.Button(control_frame, text="Analyze Current", 
                                       command=self.capture_and_analyze, bg="#ccccff")
        self.capture_button.grid(row=2, column=1, padx=5, pady=5)
        self.refresh_cameras()

    def create_doctor_results_section(self):
        doctor_frame = tk.LabelFrame(self, text="Doctor Information", font=("Helvetica", 12, "bold"))
        doctor_frame.grid(row=1, column=2, padx=10, pady=10, sticky="nsew", ipadx=10, ipady=10)
        tk.Label(doctor_frame, text="Doctor ID:").grid(row=0, column=0, sticky="w", pady=5)
        self.doctor_id_var = tk.StringVar()
        tk.Entry(doctor_frame, textvariable=self.doctor_id_var, width=20).grid(row=0, column=1, pady=5)
        tk.Label(doctor_frame, text="Doctor Name:").grid(row=1, column=0, sticky="w", pady=5)
        self.doctor_name_var = tk.StringVar()
        tk.Entry(doctor_frame, textvariable=self.doctor_name_var, width=20).grid(row=1, column=1, pady=5)
        tk.Label(doctor_frame, text="Specialization:").grid(row=2, column=0, sticky="w", pady=5)
        self.doctor_spec_var = tk.StringVar()
        spec_combo = ttk.Combobox(doctor_frame, textvariable=self.doctor_spec_var,
                                 values=["Ophthalmologist", "General Practitioner", "Specialist"], 
                                 width=17, state="readonly")
        spec_combo.grid(row=2, column=1, pady=5)
        tk.Label(doctor_frame, text="Hospital/Clinic:").grid(row=3, column=0, sticky="w", pady=5)
        self.doctor_hospital_var = tk.StringVar()
        tk.Entry(doctor_frame, textvariable=self.doctor_hospital_var, width=20).grid(row=3, column=1, pady=5)
        results_frame = tk.LabelFrame(doctor_frame, text="Detection Results")
        results_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")
        self.result_label = tk.Label(results_frame, text="No analysis performed yet", 
                                    font=("Helvetica", 10), wraplength=200)
        self.result_label.pack(pady=10)
        button_frame = tk.Frame(doctor_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        tk.Button(button_frame, text="Clear Doctor Form", command=self.clear_doctor_form,
                 bg="#ffcccc").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Send to Server", command=self.send_to_server,
                 bg="#ffffcc").pack(side=tk.LEFT, padx=5)

    def create_console(self):
        console_frame = tk.LabelFrame(self, text="System Log", font=("Helvetica", 10, "bold"))
        console_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        self.console_log = scrolledtext.ScrolledText(console_frame, width=120, height=8, state='disabled')
        self.console_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # --- GUI Logic and Application Flow (mostly unchanged) ---
    def refresh_cameras(self):
        available_cameras = []
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available_cameras.append(str(idx))
            cap.release()
        self.camera_dropdown['values'] = ["None"] + available_cameras
        if available_cameras:
            self.camera_dropdown.current(1)

    def switch_mode(self):
        self.detection_mode = self.mode_var.get()
        if self.detection_mode == "camera":
            self.camera_controls_frame.grid()
            self.image_controls_frame.grid_remove()
            self.camera_label.config(text="Camera Stream")
            self.start_button.config(text="Start Live Detection", state="normal", bg="#ccffcc")
            self.capture_button.config(text="Capture & Analyze")
            self.display_locked = False
            if self.processing_started: self.processing_started = False
        else:
            self.camera_controls_frame.grid_remove()
            self.image_controls_frame.grid()
            self.camera_label.config(text="Imported Image")
            self.start_button.config(text="Live Detection (N/A)", state="disabled", bg="#cccccc")
            self.capture_button.config(text="Analyze Image")
            if self.processing_started: self.processing_started = False
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            if self.imported_image is None:
                self.display_locked = False
                self.show_image_placeholder()
        self.log_to_console(f"Switched to {self.detection_mode} mode")

    def show_image_placeholder(self):
        placeholder = np.zeros((375, 500, 3), dtype=np.uint8)
        placeholder.fill(50)
        cv2.putText(placeholder, "No Image Imported", (140, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder, "Click 'Import Image' to select a file", (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        img = Image.fromarray(placeholder)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_frame_widget.config(image=imgtk)
        self.camera_frame_widget.image = imgtk

    def import_image(self):
        file_path = filedialog.askopenfilename(title="Select Eye Image", filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp'), ('All files', '*.*')])
        if file_path:
            try:
                self.imported_image = cv2.imread(file_path)
                if self.imported_image is None: raise ValueError("Could not load image")
                self.imported_image_path = file_path
                filename = os.path.basename(file_path)
                self.image_path_label.config(text=f"Selected: {filename}")
                self.display_imported_image()
                self.display_locked = True
                self.log_to_console(f"Image imported: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import image: {e}")
                self.log_to_console(f"Error importing image: {e}")

    def display_imported_image(self):
        if self.imported_image is not None:
            self.display_analysis_result(self.imported_image)

    def apply_selected_camera(self):
        camera_id = self.camera_var.get()
        if self.cap is not None: self.cap.release()
        if camera_id.isdigit():
            self.cap = cv2.VideoCapture(int(camera_id))
            self.log_to_console(f"Camera {camera_id} applied." if self.cap.isOpened() else f"Error: Cannot open camera {camera_id}")
        else:
            self.cap = None

    def toggle_detection(self):
        self.processing_started = not self.processing_started
        if self.processing_started:
            self.start_button.config(text="Stop Live Detection", bg="#ffcccc")
            self.log_to_console("Live detection started")
        else:
            self.start_button.config(text="Start Live Detection", bg="#ccffcc")
            self.log_to_console("Live detection stopped")

    def capture_and_analyze(self):
        if not self.validate_forms(): return
        
        if self.detection_mode == "camera":
            if self.frame is None: messagebox.showerror("Error", "No camera feed"); return
            analysis_image = self.frame.copy()
            source_info = "camera"
        else:
            if self.imported_image is None: messagebox.showerror("Error", "No image imported"); return
            analysis_image = self.imported_image.copy()
            source_info = f"image: {os.path.basename(self.imported_image_path)}"
            
        self.log_to_console(f"Analyzing {source_info}...")
        analysis_result = self.analyze_for_glaucoma(analysis_image)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        result_path = os.path.join("results", f"glaucoma_result_{timestamp}.jpg")
        cv2.imwrite(result_path, analysis_result['annotated_image'])
        
        original_path = os.path.join("results", f"glaucoma_original_{timestamp}.jpg")
        cv2.imwrite(original_path, analysis_image)
        self.original_image_path = original_path
        
        self.detection_result = analysis_result
        self.result_image = result_path
        self.update_result_display(analysis_result)
        
        if self.detection_mode == "image":
            self.display_analysis_result(analysis_result['annotated_image'])
            self.display_locked = True
        
        self.log_to_console(f"Analysis complete. Result saved: {result_path}")

    def display_analysis_result(self, image_to_display):
        display_image = cv2.cvtColor(image_to_display, cv2.COLOR_BGR2RGB)
        h, w = display_image.shape[:2]
        max_w, max_h = 500, 375
        scale = min(max_w/w, max_h/h) if w > max_w or h > max_h else 1
        new_w, new_h = int(w * scale), int(h * scale)
        display_image = cv2.resize(display_image, (new_w, new_h))
        
        img = Image.fromarray(display_image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.camera_frame_widget.config(image=imgtk)
        self.camera_frame_widget.image = imgtk

    def _get_class_name(self, cls_id):
        return self.class_names.get(int(cls_id), f"Class_{cls_id}")

    def analyze_for_glaucoma(self, frame):
        result = {'has_glaucoma': False, 'confidence': 0.0, 'detected_features': [], 'annotated_image': frame.copy()}
        try:
            detections = self.detector.infer(frame)
            annotated_frame = frame.copy()
            
            for x1, y1, x2, y2, conf, cls_id in detections:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_name = self._get_class_name(cls_id)
                
                if 'glaucoma' in class_name.lower():
                    result['has_glaucoma'] = True
                    result['confidence'] = max(result['confidence'], conf)

                result['detected_features'].append({'class': class_name, 'confidence': conf, 'bbox': [x1, y1, x2, y2]})

                color = self.colors[int(cls_id) % len(self.colors)]
                label = f"{class_name} ({conf:.2f})"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            result['annotated_image'] = annotated_frame
        except Exception as e:
            self.log_to_console(f"Error during analysis: {e}")
        return result

    def update_result_display(self, res):
        if res['has_glaucoma']:
            text = f"GLAUCOMA DETECTED\nConfidence: {res['confidence']:.2f}"
            self.result_label.config(text=text, fg="red")
        else:
            text = "NO GLAUCOMA DETECTED"
            self.result_label.config(text=text, fg="green")

    def validate_forms(self):
        if not self.patient_id_var.get() or not self.patient_name_var.get():
            messagebox.showerror("Error", "Please fill in Patient ID and Name")
            return False
        if not self.doctor_id_var.get() or not self.doctor_name_var.get():
            messagebox.showerror("Error", "Please fill in Doctor ID and Name")
            return False
        return True

    def collect_patient_info(self):
        return {'patient_id': self.patient_id_var.get(), 'name': self.patient_name_var.get(), 'age': self.patient_age_var.get(), 'gender': self.patient_gender_var.get(), 'phone': self.patient_phone_var.get(), 'medical_history': self.patient_history_text.get("1.0", tk.END).strip(), 'examination_date': datetime.now().isoformat()}

    def collect_doctor_info(self):
        return {'doctor_id': self.doctor_id_var.get(), 'name': self.doctor_name_var.get(), 'specialization': self.doctor_spec_var.get(), 'hospital': self.doctor_hospital_var.get()}

    def send_to_server(self):
        if self.detection_result is None: messagebox.showerror("Error", "No analysis to send"); return
        if not self.result_image or not self.original_image_path: messagebox.showerror("Error", "Image paths not found"); return
        
        try:
            headers = {'Cookie': self.server_cookie} if self.server_cookie else {}
            
            self.log_to_console("Uploading original image...")
            with open(self.original_image_path, 'rb') as f:
                r1 = requests.post(self.server_upload_url, files={'image': f}, headers=headers, timeout=30)
            r1.raise_for_status()
            original_url = r1.json().get('url')
            if not original_url: raise ValueError("No URL in original image upload response")
            self.log_to_console(f"Original image uploaded: {original_url}")
            
            self.log_to_console("Uploading detected image...")
            with open(self.result_image, 'rb') as f:
                r2 = requests.post(self.server_upload_url, files={'image': f}, headers=headers, timeout=30)
            r2.raise_for_status()
            detected_url = r2.json().get('url')
            if not detected_url: raise ValueError("No URL in detected image upload response")
            self.log_to_console(f"Detected image uploaded: {detected_url}")

            patient = self.collect_patient_info()
            doctor = self.collect_doctor_info()
            payload = {"patientName": patient['name'], "age": int(patient.get('age') or 0), "gender": patient['gender'].lower(), "imageDetected": detected_url, "imageOriginal": original_url, "mlHasDisease": bool(self.detection_result['has_glaucoma']), "diseaseName": "glaucoma", "createdBy": doctor['doctor_id']}

            self.log_to_console("Creating record on server...")
            r3 = requests.post(self.server_record_url, json=payload, headers={**headers, 'Content-Type': 'application/json'}, timeout=30)
            r3.raise_for_status()
            self.log_to_console("Record created successfully")
            messagebox.showinfo("Success", "Sent to server successfully")
        except Exception as e:
            self.log_to_console(f"Error sending to server: {e}")
            messagebox.showerror("Error", f"Failed to send data: {e}")

    def clear_patient_form(self):
        self.patient_id_var.set(""); self.patient_name_var.set(""); self.patient_age_var.set(""); self.patient_gender_var.set(""); self.patient_phone_var.set(""); self.patient_history_text.delete("1.0", tk.END)

    def clear_doctor_form(self):
        self.doctor_id_var.set(""); self.doctor_name_var.set(""); self.doctor_spec_var.set(""); self.doctor_hospital_var.set("")

    def update_frames(self):
        while self.camera_running:
            try:
                if self.detection_mode == "camera":
                    if self.cap and self.cap.isOpened():
                        self.ret, self.frame = self.cap.read()
                    else:
                        self.ret, self.frame = False, None

                    if not self.ret or self.frame is None:
                        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(self.frame, "No Camera Connected", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    display_frame = self.frame.copy()
                    if self.processing_started and self.ret:
                        self.run_live_detection(display_frame)

                    if not self.display_locked:
                        self.update_camera_display(display_frame)
                time.sleep(0.03)  # ~30 FPS
            except Exception as e:
                self.log_to_console(f"Error in update_frames: {e}")

    def run_live_detection(self, frame):
        try:
            start_time = time.time()
            detections = self.detector.infer(frame)
            fps = 1 / (time.time() - start_time)
            self.total_fps.append(fps)

            self.boxes = [(d[0], d[1], d[2], d[3]) for d in detections]
            self.confs = [d[4] for d in detections]
            self.clss = [d[5] for d in detections]

            for box, cls_id, score in zip(self.boxes, self.clss, self.confs):
                x1, y1, x2, y2 = map(int, box)
                class_name = self._get_class_name(cls_id)
                label = f"{class_name} ({score:.2f})"
                color = self.colors[cls_id % len(self.colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if self.total_fps:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception as e:
            self.log_to_console(f"Error in live detection: {e}")

    def update_camera_display(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (500, 375))
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_frame_widget.config(image=imgtk)
            self.camera_frame_widget.image = imgtk
        except Exception as e:
            self.log_to_console(f"Error updating camera display: {e}")

    def log_to_console(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.console_log.config(state='normal')
        self.console_log.insert(tk.END, formatted_message)
        self.console_log.config(state='disabled')
        self.console_log.see(tk.END)

    def on_closing(self):
        """Handle window closing event."""
        self.camera_running = False
        if self.update_frames_thread.is_alive():
            self.update_frames_thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
        self.destroy()

def test_bounding_boxes():
    """
    Simple test function to verify bounding box detection.
    Call this function to test with a sample image.
    """
    try:
        # Initialize detector
        detector = TRTDetector(
            "model_results/runs/train/glaucoma_yolo11/weights/best.engine",
            conf_threshold=0.25,
            iou_threshold=0.45
        )

        # Test with a sample image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        print("Testing bounding box detection...")
        detections = detector.infer(test_image)

        print(f"Found {len(detections)} detections:")
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls_id = det
            print(f"  Detection {i}: Class {cls_id}, Conf {conf:.3f}, Box [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    # Uncomment the line below to run a quick test
    # test_bounding_boxes()

    app = GlaucomaApplicationTRT()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
