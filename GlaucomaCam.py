import cv2
import numpy as np
import torch
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
from ultralytics import YOLO
import os
from datetime import datetime


class GlaucomaApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Glaucoma Detection System")
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
        self.display_locked = False  # Prevent camera updates from overwriting imported images
        
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

        # Initialize UI mode after all variables are set
        self.switch_mode()
        
        self.update_frames_thread = threading.Thread(target=self.update_frames)
        self.update_frames_thread.daemon = True
        self.update_frames_thread.start()

    def setup_model(self):
        """Initialize the YOLOv11 model for glaucoma detection"""
        # Check CUDA availability and set device
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.log_to_console(f"CUDA detected: {torch.cuda.get_device_name(0)}")
            self.log_to_console(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            self.device = 'cpu'
            self.log_to_console("CUDA not available, using CPU")
        
        # Path to trained YOLOv11 glaucoma model
        self.model_path = os.path.join("model_results", "runs", "train", "glaucoma_yolo11", "weights", "best.pt")
        
        # Fallback to pre-trained weights if trained model doesn't exist
        if not os.path.exists(self.model_path):
            self.model_path = os.path.join("model_results", "yolo11n.pt")
            if not os.path.exists(self.model_path):
                self.model_path = "yolo11n.pt"  # Download from Ultralytics if needed
            self.log_to_console("Using pre-trained weights. Train a glaucoma-specific model for better results.")
        
        try:
            # Load YOLOv11 model using Ultralytics
            self.model = YOLO(self.model_path)
            
            # Explicitly move model to GPU
            if self.device == 'cuda:0':
                self.model.to(self.device)
                self.log_to_console("Model moved to GPU")
            
            # Get class names from the model
            self.class_names = self.model.names
            self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
            
            self.log_to_console(f"YOLOv11 model loaded successfully from: {self.model_path}")
            self.log_to_console(f"Classes: {list(self.class_names.values())}")
            self.log_to_console(f"Model device: {self.device}")
            
        except Exception as e:
            self.log_to_console(f"Model loading failed: {e}")
            # Try to download and use base YOLOv11 model
            try:
                self.model = YOLO('yolo11n.pt')
                
                # Explicitly move model to GPU
                if self.device == 'cuda:0':
                    self.model.to(self.device)
                    self.log_to_console("Fallback model moved to GPU")
                
                self.class_names = self.model.names
                self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]
                self.log_to_console("Loaded base YOLOv11 model as fallback")
                self.log_to_console(f"Fallback model device: {self.device}")
            except Exception as e2:
                self.log_to_console(f"Fallback model loading also failed: {e2}")

    def create_widgets(self):
        """Create the main GUI layout"""
        # Header with logos and title
        self.create_header()
        
        # Main content area with three columns
        self.create_main_content()
        
        # Console log at bottom
        self.create_console()

    def create_header(self):
        """Create header with logos and title"""
        header_frame = tk.Frame(self)
        header_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="ew")
        
        # Title
        title_label = tk.Label(header_frame, text="Glaucoma Detection System", 
                              font=("Helvetica", 24, "bold"))
        title_label.pack(pady=10)

    def create_main_content(self):
        """Create the main content area with three columns"""
        # Left column - Patient Information
        self.create_patient_form()
        
        # Middle column - Camera and controls
        self.create_camera_section()
        
        # Right column - Doctor Information and Results
        self.create_doctor_results_section()

    def create_patient_form(self):
        """Create patient information form"""
        patient_frame = tk.LabelFrame(self, text="Patient Information", font=("Helvetica", 12, "bold"))
        patient_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew", ipadx=10, ipady=10)

        # Patient form fields
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

        # Clear patient form button
        tk.Button(patient_frame, text="Clear Form", command=self.clear_patient_form,
                 bg="#ffcccc").grid(row=6, column=0, columnspan=2, pady=10)

    def create_camera_section(self):
        """Create camera display and control section"""
        camera_frame = tk.LabelFrame(self, text="Eye Examination", font=("Helvetica", 12, "bold"))
        camera_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Mode selection
        mode_frame = tk.Frame(camera_frame)
        mode_frame.grid(row=0, column=0, pady=5)
        
        tk.Label(mode_frame, text="Detection Mode:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=5)
        
        self.mode_var = tk.StringVar(value="camera")
        tk.Radiobutton(mode_frame, text="Camera Stream", variable=self.mode_var, value="camera",
                      command=self.switch_mode).grid(row=0, column=1, padx=5)
        tk.Radiobutton(mode_frame, text="Import Image", variable=self.mode_var, value="image",
                      command=self.switch_mode).grid(row=0, column=2, padx=5)

        # Camera display
        self.camera_label = tk.Label(camera_frame, text="Camera Stream")
        self.camera_label.grid(row=1, column=0, pady=10)
        
        self.camera_frame_widget = tk.Label(camera_frame)
        self.camera_frame_widget.grid(row=2, column=0, pady=10)

        # Control buttons
        control_frame = tk.Frame(camera_frame)
        control_frame.grid(row=3, column=0, pady=10)

        # Camera controls (row 0)
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

        # Image import controls (row 1)
        self.image_controls_frame = tk.Frame(control_frame)
        self.image_controls_frame.grid(row=1, column=0, columnspan=3, pady=5)
        
        tk.Button(self.image_controls_frame, text="Import Image", command=self.import_image,
                 bg="#ffcc99").grid(row=0, column=0, padx=5, pady=5)
        
        self.image_path_label = tk.Label(self.image_controls_frame, text="No image selected", 
                                        wraplength=300, justify="left")
        self.image_path_label.grid(row=0, column=1, padx=5, pady=5)

        # Detection controls (row 2)
        self.start_button = tk.Button(control_frame, text="Start Live Detection", 
                                     command=self.toggle_detection, bg="#ccffcc")
        self.start_button.grid(row=2, column=0, padx=5, pady=5)

        self.capture_button = tk.Button(control_frame, text="Analyze Current", 
                                       command=self.capture_and_analyze, bg="#ccccff")
        self.capture_button.grid(row=2, column=1, padx=5, pady=5)

        # Initialize camera list
        self.refresh_cameras()

    def create_doctor_results_section(self):
        """Create doctor information and results section"""
        # Doctor Information
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

        # Results section
        results_frame = tk.LabelFrame(doctor_frame, text="Detection Results")
        results_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        self.result_label = tk.Label(results_frame, text="No analysis performed yet", 
                                    font=("Helvetica", 10), wraplength=200)
        self.result_label.pack(pady=10)

        # Action buttons
        button_frame = tk.Frame(doctor_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)

        tk.Button(button_frame, text="Clear Doctor Form", command=self.clear_doctor_form,
                 bg="#ffcccc").pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Send to Server", command=self.send_to_server,
                 bg="#ffffcc").pack(side=tk.LEFT, padx=5)

    def create_console(self):
        """Create console log section"""
        console_frame = tk.LabelFrame(self, text="System Log", font=("Helvetica", 10, "bold"))
        console_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="ew")

        self.console_log = scrolledtext.ScrolledText(console_frame, width=120, height=8, state='disabled')
        self.console_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def refresh_cameras(self):
        """Scan and update available cameras"""
        available_cameras = []
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                available_cameras.append(str(idx))
            cap.release()
        
        self.camera_dropdown['values'] = ["None"] + available_cameras
        if available_cameras:
            self.camera_dropdown.current(1)  # Select first available camera

    def switch_mode(self):
        """Switch between camera and image modes"""
        self.detection_mode = self.mode_var.get()
        
        if self.detection_mode == "camera":
            # Show camera controls, hide image controls
            self.camera_controls_frame.grid()
            self.image_controls_frame.grid_remove()
            self.camera_label.config(text="Camera Stream")
            self.start_button.config(text="Start Live Detection", state="normal", bg="#ccffcc")
            self.capture_button.config(text="Capture & Analyze")
            
            # Unlock display for camera updates
            self.display_locked = False
            
            # Stop live detection if running
            if self.processing_started:
                self.processing_started = False  # Reset without calling toggle_detection to avoid recursion
                
        else:  # image mode
            # Hide camera controls, show image controls
            self.camera_controls_frame.grid_remove()
            self.image_controls_frame.grid()
            self.camera_label.config(text="Imported Image")
            self.start_button.config(text="Live Detection (N/A)", state="disabled", bg="#cccccc")
            self.capture_button.config(text="Analyze Image")
            
            # Stop live detection and release camera
            if self.processing_started:
                self.processing_started = False  # Reset without calling toggle_detection
            if self.cap is not None:
                self.cap.release()
                self.cap = None
                
            # Keep display locked if image is imported, otherwise show placeholder
            if self.imported_image is None:
                self.display_locked = False
                # Show placeholder for image mode
                self.show_image_placeholder()
                
        self.log_to_console(f"Switched to {self.detection_mode} mode")

    def show_image_placeholder(self):
        """Show placeholder when in image mode with no image imported"""
        try:
            # Create placeholder image
            placeholder = np.zeros((375, 500, 3), dtype=np.uint8)
            placeholder.fill(50)  # Dark gray background
            
            # Add text
            cv2.putText(placeholder, "No Image Imported", (140, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(placeholder, "Click 'Import Image' to select a file", (90, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Convert to PhotoImage and display
            img = Image.fromarray(placeholder)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.camera_frame_widget.config(image=imgtk)
            self.camera_frame_widget.image = imgtk
            
        except Exception as e:
            self.log_to_console(f"Error showing placeholder: {e}")

    def import_image(self):
        """Import an image file for analysis"""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Eye Image for Glaucoma Detection",
            filetypes=file_types,
            initialdir=os.getcwd()
        )
        
        if file_path:
            try:
                # Load and validate image
                self.imported_image = cv2.imread(file_path)
                if self.imported_image is None:
                    raise ValueError("Could not load image file")
                
                self.imported_image_path = file_path
                
                # Update UI
                filename = os.path.basename(file_path)
                self.image_path_label.config(text=f"Selected: {filename}")
                
                # Display the imported image
                self.display_imported_image()
                
                # Lock display to prevent camera updates from overwriting
                self.display_locked = True
                
                self.log_to_console(f"Image imported successfully: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import image: {e}")
                self.log_to_console(f"Error importing image: {e}")

    def display_imported_image(self):
        """Display the imported image in the camera frame"""
        if self.imported_image is not None:
            try:
                # Convert and resize for display
                display_image = cv2.cvtColor(self.imported_image, cv2.COLOR_BGR2RGB)
                height, width = display_image.shape[:2]
                
                # Calculate resize dimensions maintaining aspect ratio
                max_width, max_height = 500, 375
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_image = cv2.resize(display_image, (new_width, new_height))
                
                # Convert to PhotoImage and display
                img = Image.fromarray(display_image)
                imgtk = ImageTk.PhotoImage(image=img)
                
                self.camera_frame_widget.config(image=imgtk)
                self.camera_frame_widget.image = imgtk
                
            except Exception as e:
                self.log_to_console(f"Error displaying imported image: {e}")

    def apply_selected_camera(self):
        """Apply the selected camera"""
        camera_id = self.camera_var.get()
        
        if self.cap is not None:
            self.cap.release()
            
        if camera_id != "None" and camera_id.isdigit():
            self.cap = cv2.VideoCapture(int(camera_id))
            if self.cap.isOpened():
                self.log_to_console(f"Camera {camera_id} applied successfully")
            else:
                self.log_to_console(f"Error: Cannot open camera {camera_id}")
                self.cap = None
        else:
            self.cap = None

    def toggle_detection(self):
        """Toggle detection on/off"""
        self.processing_started = not self.processing_started
        if self.processing_started:
            if self.detection_mode == "camera":
                self.start_button.config(text="Stop Live Detection", bg="#ffcccc")
                self.log_to_console("Live detection started")
            else:
                # Should not happen in image mode, but handle gracefully
                self.processing_started = False
                self.log_to_console("Live detection not available in image mode")
        else:
            if self.detection_mode == "camera":
                self.start_button.config(text="Start Live Detection", bg="#ccffcc")
                self.log_to_console("Live detection stopped")

    def capture_and_analyze(self):
        """Capture current frame or analyze imported image"""
        if not self.validate_forms():
            return
            
        # Get the image to analyze based on current mode
        if self.detection_mode == "camera":
            if self.frame is None:
                messagebox.showerror("Error", "No camera feed available")
                return
            analysis_image = self.frame.copy()
            source_info = "camera"
        else:  # image mode
            if self.imported_image is None:
                messagebox.showerror("Error", "No image imported")
                return
            analysis_image = self.imported_image.copy()
            source_info = f"imported image: {os.path.basename(self.imported_image_path)}"
            
        self.log_to_console(f"Performing glaucoma analysis on {source_info}...")
        
        # Process the image
        analysis_result = self.analyze_for_glaucoma(analysis_image)
        
        # Save the result image and the original image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"glaucoma_result_{timestamp}.jpg"
        result_path = os.path.join("results", result_filename)
        os.makedirs("results", exist_ok=True)
        
        cv2.imwrite(result_path, analysis_result['annotated_image'])
        original_filename = f"glaucoma_original_{timestamp}.jpg"
        original_path = os.path.join("results", original_filename)
        try:
            cv2.imwrite(original_path, analysis_image)
            self.original_image_path = original_path
        except Exception:
            self.original_image_path = self.imported_image_path if self.detection_mode == "image" else None
        
        # Update results
        self.detection_result = analysis_result
        self.result_image = result_path
        
        # Update UI
        self.update_result_display(analysis_result)
        
        # If in image mode, also display the result
        if self.detection_mode == "image":
            self.display_analysis_result(analysis_result['annotated_image'])
            self.display_locked = True  # Keep the analysis result displayed
        
        self.log_to_console(f"Analysis completed. Result saved: {result_path}")

    def display_analysis_result(self, annotated_image):
        """Display analysis result in the image frame"""
        try:
            # Convert and resize for display
            display_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            height, width = display_image.shape[:2]
            
            # Calculate resize dimensions maintaining aspect ratio
            max_width, max_height = 500, 375
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_image = cv2.resize(display_image, (new_width, new_height))
            
            # Convert to PhotoImage and display
            img = Image.fromarray(display_image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.camera_frame_widget.config(image=imgtk)
            self.camera_frame_widget.image = imgtk
            
        except Exception as e:
            self.log_to_console(f"Error displaying analysis result: {e}")

    def analyze_for_glaucoma(self, frame):
        """Analyze frame for glaucoma detection using YOLOv11"""
        result = {
            'has_glaucoma': False,
            'confidence': 0.0,
            'detected_features': [],
            'annotated_image': frame.copy()
        }
        
        try:
            # Run YOLOv11 inference with explicit device
            results = self.model(frame, conf=0.25, iou=0.7, verbose=False, device=self.device)
            
            # Process detections
            annotated_frame = frame.copy()
            
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    # Get detection data
                    boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = map(int, box)
                        class_name = self.class_names[cls_id]
                        confidence = float(conf)
                        
                        # Determine if glaucoma indicators are present
                        # For glaucoma detection, look for specific classes or high confidence detections
                        if ('glaucoma' in class_name.lower() or 
                            'optic' in class_name.lower() or 
                            'disc' in class_name.lower() or 
                            'cup' in class_name.lower() or
                            confidence > 0.6):
                            result['has_glaucoma'] = True
                            result['confidence'] = max(result['confidence'], confidence)
                        
                        result['detected_features'].append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': [x1, y1, x2, y2]
                        })
                        
                        # Draw bounding box
                        color = self.colors[cls_id % len(self.colors)]
                        label = f"{class_name} ({confidence:.2f})"
                        
                        # Draw rectangle and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Label background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                                    (x1 + label_size[0], y1), color, -1)
                        
                        # Label text
                        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            result['annotated_image'] = annotated_frame
            
        except Exception as e:
            self.log_to_console(f"Error in glaucoma analysis: {e}")
        
        return result

    def update_result_display(self, analysis_result):
        """Update the result display with analysis results"""
        if analysis_result['has_glaucoma']:
            result_text = f"GLAUCOMA DETECTED\nConfidence: {analysis_result['confidence']:.2f}\n"
            result_text += f"Features found: {len(analysis_result['detected_features'])}"
            self.result_label.config(text=result_text, fg="red")
        else:
            result_text = f"NO GLAUCOMA DETECTED\n"
            result_text += f"Features analyzed: {len(analysis_result['detected_features'])}"
            self.result_label.config(text=result_text, fg="green")

    def validate_forms(self):
        """Validate that required form fields are filled"""
        # Check patient info
        if not self.patient_id_var.get() or not self.patient_name_var.get():
            messagebox.showerror("Error", "Please fill in Patient ID and Name")
            return False
        
        # Check doctor info
        if not self.doctor_id_var.get() or not self.doctor_name_var.get():
            messagebox.showerror("Error", "Please fill in Doctor ID and Name")
            return False
        
        return True

    def collect_patient_info(self):
        """Collect patient information from form"""
        history = self.patient_history_text.get("1.0", tk.END).strip()
        return {
            'patient_id': self.patient_id_var.get(),
            'name': self.patient_name_var.get(),
            'age': self.patient_age_var.get(),
            'gender': self.patient_gender_var.get(),
            'phone': self.patient_phone_var.get(),
            'medical_history': history,
            'examination_date': datetime.now().isoformat()
        }

    def collect_doctor_info(self):
        """Collect doctor information from form"""
        return {
            'doctor_id': self.doctor_id_var.get(),
            'name': self.doctor_name_var.get(),
            'specialization': self.doctor_spec_var.get(),
            'hospital': self.doctor_hospital_var.get()
        }

    def send_to_server(self):
        """Send results to server: upload images, then create record"""
        if self.detection_result is None:
            messagebox.showerror("Error", "No analysis results to send")
            return
        
        try:
            if not self.result_image:
                messagebox.showerror("Error", "No result image found")
                return

            if not self.original_image_path and self.imported_image_path:
                self.original_image_path = self.imported_image_path

            if not self.original_image_path:
                messagebox.showerror("Error", "No original image available to upload")
                return

            headers = {}
            if self.server_cookie:
                headers['Cookie'] = self.server_cookie

            self.log_to_console("Uploading original image...")
            self.log_to_console(f"Upload URL: {self.server_upload_url}")
            self.log_to_console(f"Original image path: {self.original_image_path}")
            
            with open(self.original_image_path, 'rb') as f:
                files = {'image': (os.path.basename(self.original_image_path), f, 'image/jpeg')}
                r1 = requests.post(self.server_upload_url, files=files, headers=headers, timeout=30)
            
            self.log_to_console(f"Upload response status: {r1.status_code}")
            self.log_to_console(f"Upload response text: {r1.text}")
            r1.raise_for_status()
            
            response_json = r1.json()
            original_url = response_json.get('url')
            if not original_url:
                raise ValueError(f"No URL in upload response: {response_json}")

            self.log_to_console(f"Original image uploaded: {original_url}")
            self.log_to_console("Uploading detected image...")
            
            with open(self.result_image, 'rb') as f:
                files = {'image': (os.path.basename(self.result_image), f, 'image/jpeg')}
                r2 = requests.post(self.server_upload_url, files=files, headers=headers, timeout=30)
            
            self.log_to_console(f"Detected upload response status: {r2.status_code}")
            r2.raise_for_status()
            
            response_json = r2.json()
            detected_url = response_json.get('url')
            if not detected_url:
                raise ValueError(f"No URL in detected upload response: {response_json}")
            
            self.log_to_console(f"Detected image uploaded: {detected_url}")

            patient = self.collect_patient_info()
            doctor = self.collect_doctor_info()
            try:
                age_val = int(patient.get('age') or 0)
            except Exception:
                age_val = 0
            gender_val = (patient.get('gender') or '').strip().lower()

            payload = {
                "patientName": patient.get('name') or "",
                "age": age_val,
                "gender": gender_val,
                "imageDetected": detected_url,
                "imageOriginal": original_url,
                "mlHasDisease": bool(self.detection_result.get('has_glaucoma')),
                "diseaseName": "glaucoma",
                "createdBy": doctor.get('doctor_id') or ""
            }

            self.log_to_console("Creating record on server...")
            r3 = requests.post(self.server_record_url, json=payload, headers={**headers, 'Content-Type': 'application/json'}, timeout=30)
            r3.raise_for_status()

            self.log_to_console("Record created successfully")
            messagebox.showinfo("Success", "Sent to server successfully")
            
        except Exception as e:
            self.log_to_console(f"Error sending to server: {e}")
            messagebox.showerror("Error", f"Failed to send data: {e}")

    def clear_patient_form(self):
        """Clear patient information form"""
        self.patient_id_var.set("")
        self.patient_name_var.set("")
        self.patient_age_var.set("")
        self.patient_gender_var.set("")
        self.patient_phone_var.set("")
        self.patient_history_text.delete("1.0", tk.END)

    def clear_doctor_form(self):
        """Clear doctor information form"""
        self.doctor_id_var.set("")
        self.doctor_name_var.set("")
        self.doctor_spec_var.set("")
        self.doctor_hospital_var.set("")

    def update_frames(self):
        """Main camera update loop"""
        while self.camera_running:
            try:
                # Only process camera frames in camera mode
                if self.detection_mode == "camera":
                    if self.cap is not None and self.cap.isOpened():
                        self.ret, self.frame = self.cap.read()
                    else:
                        self.ret, self.frame = False, None

                    # Create blank frame if no camera
                    if not self.ret or self.frame is None:
                        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(self.frame, "No Camera Connected", (180, 240), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                    display_frame = self.frame.copy()

                    # Run detection if started and we have a valid frame
                    if self.processing_started and self.ret:
                        self.run_live_detection(display_frame)

                    # Update camera display only in camera mode and when not locked
                    if not self.display_locked:
                        self.update_camera_display(display_frame)
                
                # In image mode, don't update the display (keep imported image)
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                self.log_to_console(f"Error in update_frames: {e}")

    def run_live_detection(self, frame):
        """Run live detection on frame using YOLOv11"""
        try:
            # Clear previous detections
            self.boxes.clear()
            self.confs.clear()
            self.clss.clear()

            # Run YOLOv11 inference with explicit device
            start_time = time.time()
            results = self.model(frame, conf=0.25, iou=0.7, verbose=False, device=self.device)
            fps = 1 / (time.time() - start_time)
            self.total_fps.append(fps)

            # Process detections
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    # Get detection data
                    boxes = r.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                    confidences = r.boxes.conf.cpu().numpy()
                    class_ids = r.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        self.confs.append(float(conf))
                        self.clss.append(int(cls_id))
                        self.boxes.append((x1, y1, x2, y2))

            # Draw detections on frame
            for box, cls_id, score in zip(self.boxes, self.clss, self.confs):
                x1, y1, x2, y2 = box
                class_name = self.class_names[cls_id]
                label = f"{class_name} ({score:.2f})"
                color = self.colors[cls_id % len(self.colors)]
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display FPS
            if self.total_fps:
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            self.log_to_console(f"Error in live detection: {e}")

    def update_camera_display(self, frame):
        """Update camera display widget"""
        try:
            # Convert and resize frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (500, 375))
            
            # Convert to PhotoImage
            img = Image.fromarray(frame_resized)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update display
            self.camera_frame_widget.config(image=imgtk)
            self.camera_frame_widget.image = imgtk
            
        except Exception as e:
            self.log_to_console(f"Error updating camera display: {e}")

    def log_to_console(self, message):
        """Add message to console log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.console_log.config(state='normal')
        self.console_log.insert(tk.END, formatted_message + "\n")
        self.console_log.config(state='disabled')
        self.console_log.see(tk.END)

    def __del__(self):
        """Cleanup when application closes"""
        self.camera_running = False
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    app = GlaucomaApplication()
    app.mainloop() 