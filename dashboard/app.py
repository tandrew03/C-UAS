import tkinter as tk
from tkinter import ttk, filedialog
from datetime import datetime

import cv2
from PIL import Image, ImageTk

# Install dependencies first:
# python -m pip install opencv-python pillow

# Set this to the camera you want to use:
# 0 = first camera, 1 = second camera, etc.
CAMERA_INDEX = 0

# Color Palette
COLORS = {
    "background": "#0F172A", # Deep navy application background
    "panel": "#1E293B", # Slate blue-gray panels
    "camera": "#020617", # Near-black camera feed
    "border": "#334155", # Muted slate borders and progress track
    "text": "#E2E8F0", # Off-white primary text
    "muted_text": "#94A3B8", # Cool gray secondary text
    "blue": "#2563EB", # Primary control blue
    "blue_hover": "#3B82F6", # Button hover blue
    "green": "#22C55E", # Ready/healthy status
    "tracking": "#84CC16", # Bounding-box color
    "amber": "#F59E0B", # Warning state
    "alert": "#EF4444", # Alert/error state
}


# Main Camera tracking window
class CameraTrackingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Camera Feed")
        self.geometry("900x560")
        self.minsize(760, 480)
        self.configure(bg=COLORS["background"])
        self.protocol("WM_DELETE_WINDOW", self.close_app)

        self.paused = False
        self.confidence_score = tk.DoubleVar(value=0.88)
        self.box_x_ratio = 0.5
        self.box_y_ratio = 0.5
        self.camera = None
        self.current_frame = None
        self.current_photo = None

        self._build_styles()
        self._build_ui()
        self._start_camera()
# Visual Styles
    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Panel.TFrame", background=COLORS["panel"], borderwidth=1, relief="flat")
        style.configure("PanelLabel.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Arial", 9))
        style.configure("Heading.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Arial", 9, "bold"))
        style.configure("Control.TButton", background=COLORS["blue"], foreground="white", borderwidth=0, padding=(8, 3), font=("Arial", 9))
        style.map("Control.TButton", background=[("active", COLORS["blue_hover"])])
# Return Stylized Panel
    def _panel(self, parent, padding=6):
        return ttk.Frame(parent, style="Panel.TFrame", padding=padding)

    def _metric_panel(self, parent, title, metrics):
        panel = self._panel(parent, 6)
        ttk.Label(panel, text=title, style="Heading.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        for row, (label, value) in enumerate(metrics, start=1):
            ttk.Label(panel, text=label, style="PanelLabel.TLabel").grid(row=row, column=0, sticky="w", pady=1)
            ttk.Label(panel, text=value, style="PanelLabel.TLabel").grid(row=row, column=1, sticky="e", pady=1)
        panel.columnconfigure(1, weight=1)
        return panel
# Create and arrange all widgets in app
    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(5, weight=1)

        tk.Label(
            self, text="Camera Feed", bg=COLORS["background"], fg=COLORS["text"],
            font=("Arial", 22), anchor="w"
        ).grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
# The workspace contains the camera feed and telemetry panels.
        workspace = tk.Frame(self, bg=COLORS["background"])
        workspace.grid(row=1, column=0, sticky="nsew", padx=10)
        workspace.columnconfigure(0, weight=3)
        workspace.columnconfigure(1, weight=2)
        workspace.rowconfigure(0, weight=1)

        camera_panel = self._panel(workspace, 0)
        camera_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        camera_panel.rowconfigure(0, weight=1)
        camera_panel.columnconfigure(0, weight=1)

        self.camera_canvas = tk.Canvas(camera_panel, bg=COLORS["camera"], highlightthickness=0)
        self.camera_canvas.grid(row=0, column=0, sticky="nsew")
        self.camera_canvas.bind("<Configure>", self._draw_camera)
        self.camera_canvas.bind("<Motion>", self._on_camera_motion)

        telemetry = tk.Frame(workspace, bg=COLORS["background"])
        telemetry.grid(row=0, column=1, sticky="nsew")
        telemetry.columnconfigure(0, weight=1)
        # Acoustic sensor information.
        self._metric_panel(
            telemetry, "Acoustic Branch",
            [("Bearing", "184.6°"), ("Drone Probability", "0.91"), ("SNR", "18.3 dB")]
        ).grid(row=0, column=0, sticky="ew", pady=(0, 5))

        # Keep Visual Branch coordinates linked to the bounding-box position
        self.bounding_box_position = tk.StringVar(value=self._box_position_text())

# Visual tracking panel
        visual_panel = self._panel(telemetry, 6)
        visual_panel.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(visual_panel, text="Visual Branch", style="Heading.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(visual_panel, text="Bounding Box (x,y)", style="PanelLabel.TLabel").grid(row=1, column=0, sticky="w", pady=1)
        ttk.Label(visual_panel, textvariable=self.bounding_box_position, style="PanelLabel.TLabel").grid(row=1, column=1, sticky="e", pady=1)
        ttk.Label(visual_panel, text="Visual Confidence", style="PanelLabel.TLabel").grid(row=2, column=0, sticky="w", pady=1)
        self.confidence_display = tk.StringVar(value="0.88")
        ttk.Label(
            visual_panel,
            textvariable=self.confidence_display,
            style="PanelLabel.TLabel",
        ).grid(row=2, column=1, sticky="e", pady=1)
        ttk.Label(visual_panel, text="Detections", style="PanelLabel.TLabel").grid(row=3, column=0, sticky="w", pady=1)
        ttk.Label(visual_panel, text="1", style="PanelLabel.TLabel").grid(row=3, column=1, sticky="e", pady=1)
        visual_panel.columnconfigure(1, weight=1)

        # Predictive tracking information
        self._metric_panel(
            telemetry, "Predictive Tracking",
            [("Predicted Position", "(0.64, 0.41)"), ("Velocity", "2.1")]
        ).grid(row=2, column=0, sticky="ew", pady=(0, 5))

        # Operator control panel
        controls = self._panel(telemetry, 6)
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        ttk.Label(controls, text="Operator Controls", style="Heading.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        self.pause_button = ttk.Button(controls, text="Pause", style="Control.TButton", command=self.toggle_pause)
        self.pause_button.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(controls, text="Reset Track", style="Control.TButton", command=self.reset_track).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Button(controls, text="Center View", style="Control.TButton", command=self.center_view).grid(row=2, column=0, sticky="ew", padx=(0, 5), pady=2)
        ttk.Button(controls, text="Export Log", style="Control.TButton", command=self.export_log).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Separator(self, orient="horizontal").grid(row=2, column=0, sticky="ew", padx=10, pady=6)

        # Bottom status row
        status_row = tk.Frame(self, bg=COLORS["background"])
        status_row.grid(row=3, column=0, sticky="ew", padx=10)
        status_row.columnconfigure(0, weight=1)
        status_row.columnconfigure(1, weight=2)
        status_row.columnconfigure(2, weight=1)
        self._status_panel(status_row, "System Status", "Ready", 0, COLORS["green"])
        self._confidence_panel(status_row)
        self._alert_button(status_row)

        alert_panel = self._panel(self, 6)
        alert_panel.grid(row=4, column=0, sticky="ew", padx=10, pady=(6, 0))

        ttk.Label(
            alert_panel,
            text="Alert List",
            style="Heading.TLabel",
        ).pack(anchor="w")

        self.alert_list = tk.Listbox(
            alert_panel,
            height=4,
            bg=COLORS["camera"],
            fg=COLORS["text"],
            selectbackground=COLORS["blue"],
            selectforeground="white",
            highlightthickness=0,
            relief="flat",
            borderwidth=0,
            font=("Arial", 9),
        )
        self.alert_list.pack(fill="both", expand=True)

        # System log panel
        log_panel = self._panel(self, 6)
        log_panel.grid(row=5, column=0, sticky="nsew", padx=10, pady=(6, 10))
        ttk.Label(log_panel, text="System Log", style="Heading.TLabel").pack(anchor="w")
        self.log = tk.Text(
            log_panel, height=4, bg=COLORS["camera"], fg=COLORS["muted_text"],
            insertbackground=COLORS["text"], relief="flat", borderwidth=0, font=("Arial", 9)
        )
        self.log.pack(fill="both", expand=True)
        self._write_log("Tracking initialized")
        self._write_log("Waiting for camera...")

    # Fusion Layer Panel
        self.fusion_score = tk.StringVar(value="0.00")
        self.fusion_state = tk.StringVar(value="PASSIVE_SCANNING")

        fusion_panel = self._panel(telemetry, 6)
        fusion_panel.grid(row=3, column=0, sticky="ew", pady=(0, 5))
        ttk.Label(fusion_panel, text="Sensor Fusion Layer", style="Heading.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        ttk.Label(fusion_panel, text="Fused Confidence", style="PanelLabel.TLabel").grid(row=1, column=0, sticky="w", pady=1)
        ttk.Label(fusion_panel, textvariable=self.fusion_score, style="PanelLabel.TLabel").grid(row=1, column=1, sticky="e", pady=1)
        ttk.Label(fusion_panel, text="Operational State", style="PanelLabel.TLabel").grid(row=2, column=0, sticky="w", pady=1)
        ttk.Label(fusion_panel, textvariable=self.fusion_state, style="PanelLabel.TLabel").grid(row=2, column=1, sticky="e", pady=1)
        fusion_panel.columnconfigure(1, weight=1)

    def _start_camera(self):
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            self._write_log(f"Unable to open camera index {CAMERA_INDEX}")
            self.camera.release()
            self.camera = None
            self._draw_camera()
            return

        self._write_log(f"Camera {CAMERA_INDEX} connected")
        self._update_camera()

    def _update_camera(self):
        if self.camera is not None and not self.paused:
            success, frame = self.camera.read()
            if success:
                self.current_frame = frame
                self._draw_camera()
            else:
                self._write_log("Could not read a camera frame")

        self.after(30, self._update_camera)  # approximately 33 FPS

    # Redraw the camera panel and place the box at the stored position
    def _draw_camera(self, event=None):
        self.camera_canvas.delete("all")
        width = self.camera_canvas.winfo_width()
        height = self.camera_canvas.winfo_height()
        if width <= 1 or height <= 1:
            return

        if self.current_frame is not None:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            # Resize the live feed to exactly match the camera panel dimensions.
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            self.current_photo = ImageTk.PhotoImage(image=image)
            self.camera_canvas.create_image(0, 0, image=self.current_photo, anchor="nw")
        else:
            self.camera_canvas.create_text(
                width / 2, height / 2,
                text="No camera feed\nCheck the camera connection and CAMERA_INDEX",
                fill=COLORS["muted_text"], justify="center", font=("Arial", 11)
            )

        box_w, box_h = 102, 52
        center_x = self.box_x_ratio * width
        center_y = self.box_y_ratio * height
        x1 = center_x - box_w / 2
        y1 = center_y - box_h / 2
        self.camera_canvas.create_rectangle(
            x1, y1, x1 + box_w, y1 + box_h,
            fill=COLORS["tracking"], outline=""
        )
        self.camera_canvas.create_text(
            center_x, center_y, text="Bounding Box",
            fill=COLORS["camera"], font=("Arial", 9, "bold")
        )
        self._update_box_position_display()

    # Create a status panel with a colored value indicator
    def _status_panel(self, parent, label, value, column, status_color):
        panel = self._panel(parent, 6)
        panel.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 5, 5 if column < 2 else 0))
        panel.columnconfigure(1, weight=1)
        ttk.Label(panel, text=label, style="PanelLabel.TLabel").grid(row=0, column=0, sticky="w")
        tk.Label(panel, text=value, bg=status_color, fg=COLORS["camera"], font=("Arial", 9, "bold")).grid(row=0, column=1, sticky="ew", padx=(8, 0))
    
    # Create the temporary red ALERT trigger button.
    def _alert_button(self, parent):
        
        panel = self._panel(parent, 6)
        panel.grid(row=0, column=2, sticky="ew", padx=(5, 0))
        panel.columnconfigure(1, weight=1)

        ttk.Label(
            panel,
            text="Alert Status",
            style="PanelLabel.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.alert_button = tk.Button(
            panel,
            text="ALERT",
            command=self.trigger_alert,
            bg=COLORS["alert"],
            activebackground="#DC2626",
            fg="white",
            activeforeground="white",
            relief="flat",
            borderwidth=0,
            font=("Arial", 9, "bold"),
            cursor="hand2",
        )
        self.alert_button.grid(row=0, column=1, sticky="ew", padx=(8, 0))

# Create a draggable confidence-score slider.
    def _confidence_panel(self, parent):
        
        panel = self._panel(parent, 6)
        panel.grid(row=0, column=1, sticky="ew", padx=5)
        panel.columnconfigure(1, weight=1)

        ttk.Label(
            panel,
            text="Confidence Score",
            style="PanelLabel.TLabel",
        ).grid(row=0, column=0, sticky="w")

        self.confidence_label = ttk.Label(
            panel,
            text="0.88",
            style="PanelLabel.TLabel",
        )
        self.confidence_label.grid(row=0, column=2, sticky="e", padx=(6, 0))

        self.confidence_slider = tk.Scale(
            panel,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            variable=self.confidence_score,
            showvalue=False,
            bg=COLORS["green"],
            fg=COLORS["text"],
            troughcolor=COLORS["border"],
            activebackground=COLORS["green"],
            highlightthickness=0,
            bd=0,
            sliderrelief="flat",
            sliderlength=14,
            command=self._on_confidence_change,
        )
        self.confidence_slider.grid(row=0, column=1, sticky="ew", padx=(10, 0))

   # Update both confidence displays when the slider moves.
    def _on_confidence_change(self, value):
        
        score = float(value)
        formatted_score = f"{score:.2f}"
        self.confidence_label.configure(text=formatted_score)
        self.confidence_display.set(formatted_score)

        # placeholder fusion layer score values
        score_acoustic = 0.85
        score_visual = score

        weight_acoustic = 0.25
        weight_visual = 0.75
        confidence = (weight_acoustic * score_acoustic) + (weight_visual * score_visual)

        self.fusion_score.set(f"{confidence:.2f}")

        if confidence < 0.40:
            self.fusion_state.set("PASSIVE_SCANNING")
        elif 0.40 <= confidence < 0.75:
            self.fusion_state.set("ACOUSTIC_DETECTION")
        else:
            self.fusion_state.set("TARGET_ALERT")

    # Return the bounding-box center as normalized x,y coordinates
    def _box_position_text(self):
        return f"({self.box_x_ratio:.2f}, {self.box_y_ratio:.2f})"

    # Update Visual Branch coordinates to match the camera-feed box
    def _update_box_position_display(self):
        self.bounding_box_position.set(self._box_position_text())

    # Move the bounding-box center to the current mouse position
    def _on_camera_motion(self, event):
        width = self.camera_canvas.winfo_width()
        height = self.camera_canvas.winfo_height()
        # Convert pixel coordinates to normalized values from 0.00 to 1.00.

        if width > 0 and height > 0:
            self.box_x_ratio = max(0.0, min(1.0, event.x / width))
            self.box_y_ratio = max(0.0, min(1.0, event.y / height))
            self._draw_camera()

    # Append a timestamped message to the system log
    def _write_log(self, message):
        if hasattr(self, "log"):
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log.insert("end", f"{timestamp} — {message}\n")
            self.log.see("end")

    # Add a timestamped detection using the current confidence score."""
    def trigger_alert(self):
        timestamp = datetime.now().strftime("%H:%M:%S")
        score = self.confidence_score.get()
        detection = f"{timestamp}  —  Detection  |  Confidence: {score:.2f}"

        self.alert_list.insert("end", detection)
        self.alert_list.see("end")
        self._write_log(f"Alert triggered — confidence {score:.2f}")
    
    # Toggle tracking between paused and active states
    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self._write_log("Camera paused" if self.paused else "Camera resumed")

    # Reset the bounding box to the center and record the action
    def reset_track(self):
        self.box_x_ratio = 0.5
        self.box_y_ratio = 0.5
        self.confidence_score.set(0.88)
        self._draw_camera()
        self._write_log("Track reset")

    # Move the bounding box to the center of the camera feed
    def center_view(self):
        self.box_x_ratio = 0.5
        self.box_y_ratio = 0.5
        self._draw_camera()
        self._write_log("View centered")

    # Save the current system log to a user-selected text file
    def export_log(self):
        path = filedialog.asksaveasfilename(
            title="Export System Log", defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if path:
            with open(path, "w", encoding="utf-8") as file:
                file.write(self.log.get("1.0", "end"))
            self._write_log("Log exported")

    def close_app(self):
        if self.camera is not None:
            self.camera.release()
        self.destroy()


if __name__ == "__main__":
    app = CameraTrackingApp()
    app.mainloop()