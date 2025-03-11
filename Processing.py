import sys
import cv2
import numpy as np
import pandas as pd
import colorsys
import time
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QRadioButton, QButtonGroup, QStatusBar, QScrollArea,
                             QApplication)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap


def get_dominant_colors(image, n_colors=10):
    small_image = cv2.resize(image, (150, 150))
    pixels = small_image.reshape(-1, 3)
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    _, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]
    centers = centers[sorted_indices]

    return centers


def generate_color_combinations(r, g, b):
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

    combinations = {
        'complementary': [],
        'analogous': [],
        'triadic': [],
        'split_complementary': [],
        'tetradic': [],
        'monochromatic': []
    }

    # Complementary
    h_comp = (h + 0.5) % 1.0
    rgb_comp = colorsys.hsv_to_rgb(h_comp, s, v)
    combinations['complementary'].append([
        (r, g, b),
        (int(rgb_comp[0] * 255), int(rgb_comp[1] * 255), int(rgb_comp[2] * 255))
    ])

    # Analogous
    h_analog1 = (h + 0.083) % 1.0
    h_analog2 = (h - 0.083) % 1.0
    rgb_analog1 = colorsys.hsv_to_rgb(h_analog1, s, v)
    rgb_analog2 = colorsys.hsv_to_rgb(h_analog2, s, v)
    combinations['analogous'].append([
        (r, g, b),
        (int(rgb_analog1[0] * 255), int(rgb_analog1[1] * 255), int(rgb_analog1[2] * 255)),
        (int(rgb_analog2[0] * 255), int(rgb_analog2[1] * 255), int(rgb_analog2[2] * 255))
    ])

    # Triadic
    h_tri1 = (h + 0.333) % 1.0
    h_tri2 = (h + 0.667) % 1.0
    rgb_tri1 = colorsys.hsv_to_rgb(h_tri1, s, v)
    rgb_tri2 = colorsys.hsv_to_rgb(h_tri2, s, v)
    combinations['triadic'].append([
        (r, g, b),
        (int(rgb_tri1[0] * 255), int(rgb_tri1[1] * 255), int(rgb_tri1[2] * 255)),
        (int(rgb_tri2[0] * 255), int(rgb_tri2[1] * 255), int(rgb_tri2[2] * 255))
    ])

    # Split-complementary
    h_split1 = (h_comp + 0.083) % 1.0
    h_split2 = (h_comp - 0.083) % 1.0
    rgb_split1 = colorsys.hsv_to_rgb(h_split1, s, v)
    rgb_split2 = colorsys.hsv_to_rgb(h_split2, s, v)
    combinations['split_complementary'].append([
        (r, g, b),
        (int(rgb_split1[0] * 255), int(rgb_split1[1] * 255), int(rgb_split1[2] * 255)),
        (int(rgb_split2[0] * 255), int(rgb_split2[1] * 255), int(rgb_split2[2] * 255))
    ])

    # Tetradic
    h_tetra1 = (h + 0.25) % 1.0
    h_tetra2 = (h + 0.5) % 1.0
    h_tetra3 = (h + 0.75) % 1.0
    rgb_tetra1 = colorsys.hsv_to_rgb(h_tetra1, s, v)
    rgb_tetra2 = colorsys.hsv_to_rgb(h_tetra2, s, v)
    rgb_tetra3 = colorsys.hsv_to_rgb(h_tetra3, s, v)
    combinations['tetradic'].append([
        (r, g, b),
        (int(rgb_tetra1[0] * 255), int(rgb_tetra1[1] * 255), int(rgb_tetra1[2] * 255)),
        (int(rgb_tetra2[0] * 255), int(rgb_tetra2[1] * 255), int(rgb_tetra2[2] * 255)),
        (int(rgb_tetra3[0] * 255), int(rgb_tetra3[1] * 255), int(rgb_tetra3[2] * 255))
    ])

    # Monochromatic
    for i in range(4):
        new_s = max(0.1, s * (0.5 + i * 0.15))
        new_v = min(1.0, v * (0.5 + i * 0.15))
        rgb_mono = colorsys.hsv_to_rgb(h, new_s, new_v)
        combinations['monochromatic'].append(
            (int(rgb_mono[0] * 255), int(rgb_mono[1] * 255), int(rgb_mono[2] * 255))
        )

    return combinations


def create_combinations_display(combinations):
    if not combinations:
        return None

    # Calculate total height needed
    row_height = 100  # Increased height for better visibility
    button_height = 30  # Height for copy button
    padding = 20  # Padding between rows
    total_height = (row_height + button_height + padding) * len(combinations)

    # Create image with white background
    combinations_image = np.zeros((total_height, 400, 3), dtype=np.uint8)
    combinations_image.fill(255)

    y_offset = 0
    color_data = {}  # Store color data for copy functionality

    for combo_type, combo_list in combinations.items():
        if combo_type == 'monochromatic':
            colors = combo_list
        else:
            colors = combo_list[0]

        # Draw combination type text
        cv2.putText(combinations_image,
                    combo_type.replace('_', ' ').title(),
                    (10, y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2)

        # Draw color rectangles
        x_offset = 0
        rect_width = 400 // len(colors)
        color_codes = []
        for color in colors:
            cv2.rectangle(combinations_image,
                          (x_offset, y_offset + 40),
                          (x_offset + rect_width, y_offset + row_height - 10),
                          (int(color[2]), int(color[1]), int(color[0])),
                          -1)
            color_codes.append(f"RGB({color[0]}, {color[1]}, {color[2]})")
            x_offset += rect_width

        # Store color codes for copy functionality
        color_data[combo_type] = ', '.join(color_codes)
        y_offset += row_height + button_height + padding

    return combinations_image, color_data


class ColorDetectionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAVE")
        self.setGeometry(100, 100, 1400, 900)

        # Initialize variables
        self.df = pd.read_csv('colors.csv', names=['color', 'color_name', 'hex', 'R', 'G', 'B'], header=None)
        self.cap = None
        self.current_frame = None
        self.original_frame = None
        self.is_video_mode = False
        self.is_paused = False
        self.current_color = (0, 0, 0)
        self.dynamic_color = (0, 0, 0)
        self.zoom_radius = 40
        self.mouse_pos = (0, 0)
        self.dominant_colors = None
        self.last_palette_update = 0
        self.palette_update_interval = 500
        self.current_combinations = None
        self.color_data = {}

        # Setup UI
        self.setup_ui()

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(33)

    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Control panel
        control_panel = QHBoxLayout()

        # Mode selection
        mode_group = QButtonGroup(self)
        self.image_mode_btn = QRadioButton("Image Mode")
        self.video_mode_btn = QRadioButton("Video Mode")
        mode_group.addButton(self.image_mode_btn)
        mode_group.addButton(self.video_mode_btn)
        self.image_mode_btn.setChecked(True)

        # Buttons
        self.file_btn = QPushButton("Open File")
        self.webcam_btn = QPushButton("Open Webcam")
        self.capture_btn = QPushButton("Capture")
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.capture_btn.setEnabled(False)

        # Add controls to panel
        control_panel.addWidget(self.image_mode_btn)
        control_panel.addWidget(self.video_mode_btn)
        control_panel.addWidget(self.file_btn)
        control_panel.addWidget(self.webcam_btn)
        control_panel.addWidget(self.capture_btn)
        control_panel.addWidget(self.pause_btn)

        # Main content area
        content_layout = QHBoxLayout()

        # Left side - Image display
        left_panel = QVBoxLayout()

        # Create scroll area for image
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(800, 600)

        # Display area
        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scroll_area.setWidget(self.display_label)

        # Palette display
        self.palette_label = QLabel()
        self.palette_label.setMinimumHeight(50)
        self.palette_label.setMaximumHeight(50)

        left_panel.addWidget(scroll_area)
        left_panel.addWidget(self.palette_label)

        # Right side - Color information and combinations
        right_panel = QVBoxLayout()

        # Color info panel
        self.color_info_label = QLabel()
        self.color_info_label.setStyleSheet("QLabel { padding: 10px; border: 1px solid #ccc; border-radius: 5px; }")
        self.color_info_label.setMaximumHeight(100)

        # Dynamic color panel
        self.dynamic_color_label = QLabel()
        self.dynamic_color_label.setStyleSheet("QLabel { padding: 10px; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px; }")
        self.dynamic_color_label.setMinimumHeight(80)

        # Color combinations panel with buttons
        combinations_widget = QWidget()
        self.combinations_layout = QVBoxLayout(combinations_widget)

        # Create scroll area for combinations
        combinations_scroll = QScrollArea()
        combinations_scroll.setWidgetResizable(True)
        combinations_scroll.setFixedSize(420, 600)
        combinations_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #f9f9f9;
                padding: 10px;
            }
            QScrollBar:vertical {
                border: none;
                background: #e0e0e0;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #a0a0a0;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Combinations container widget
        combinations_container = QWidget()
        combinations_container_layout = QVBoxLayout(combinations_container)
        combinations_container_layout.setSpacing(10)

        # Combinations display
        self.combinations_label = QLabel()
        self.combinations_label.setMinimumWidth(400)
        self.combinations_label.setStyleSheet("QLabel { background-color: white; border: 1px solid #ddd; border-radius: 3px; padding: 5px; }")

        # Copy buttons container
        self.copy_buttons_layout = QVBoxLayout()
        self.copy_buttons_layout.setSpacing(5)

        combinations_container_layout.addWidget(self.combinations_label)
        combinations_container_layout.addLayout(self.copy_buttons_layout)
        combinations_container_layout.addStretch()

        # Set the container as the scroll area's widget
        combinations_scroll.setWidget(combinations_container)

        # Add scroll area to combinations layout
        self.combinations_layout.addWidget(combinations_scroll)

        right_panel.addWidget(self.color_info_label)
        right_panel.addWidget(self.dynamic_color_label)
        right_panel.addWidget(combinations_widget)
        right_panel.addStretch()

        # Add panels to content layout
        content_layout.addLayout(left_panel, stretch=2)
        content_layout.addLayout(right_panel, stretch=1)

        # Add everything to main layout
        main_layout.addLayout(control_panel)
        main_layout.addLayout(content_layout)

        # Status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Connect signals
        self.file_btn.clicked.connect(self.open_file)
        self.webcam_btn.clicked.connect(self.open_webcam)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.capture_btn.clicked.connect(self.capture_frame)
        self.image_mode_btn.toggled.connect(self.change_mode)

        # Mouse tracking
        self.display_label.setMouseTracking(True)
        self.display_label.mousePressEvent = self.get_color
        self.display_label.mouseMoveEvent = self.update_mouse_position

    def copy_color_codes(self, combo_type):
        if combo_type in self.color_data:
            clipboard = QApplication.clipboard()
            clipboard.setText(self.color_data[combo_type])
            self.statusBar.showMessage(f"Copied {combo_type} color codes to clipboard", 2000)

    def create_zoomed_cursor(self, frame, x, y):
        height, width = frame.shape[:2]
        zoom_size = self.zoom_radius * 2
        y1 = max(0, y - self.zoom_radius)
        y2 = min(height, y + self.zoom_radius)
        x1 = max(0, x - self.zoom_radius)
        x2 = min(width, x + self.zoom_radius)

        if y2 - y1 > 0 and x2 - x1 > 0:
            zoomed_region = frame[y1:y2, x1:x2].copy()
            if zoomed_region.size > 0:
                small_size = 8
                pixelated = cv2.resize(zoomed_region, (small_size, small_size), interpolation=cv2.INTER_LINEAR)
                pixelated = cv2.resize(pixelated, (zoom_size * 2, zoom_size * 2), interpolation=cv2.INTER_NEAREST)

                cell_size = zoom_size * 2 // small_size
                for i in range(0, zoom_size * 2, cell_size):
                    cv2.line(pixelated, (i, 0), (i, zoom_size * 2), (128, 128, 128), 1)
                    cv2.line(pixelated, (0, i), (zoom_size * 2, i), (128, 128, 128), 1)

                center = zoom_size
                cv2.line(pixelated, (center - 5, center), (center + 5, center), (0, 0, 0), 2)
                cv2.line(pixelated, (center, center - 5), (center, center + 5), (0, 0, 0), 2)

                return pixelated
        return None

    def update_mouse_position(self, event):
        if self.current_frame is not None:
            x = min(max(0, event.pos().x()), self.current_frame.shape[1] - 1)
            y = min(max(0, event.pos().y()), self.current_frame.shape[0] - 1)
            self.mouse_pos = (x, y)
            self.dynamic_color = tuple(self.current_frame[y, x][::-1])
            self.update_dynamic_color_info()
            self.update_display()

    def update_dynamic_color_info(self):
        if self.current_frame is not None:
            r, g, b = [int(x) for x in self.dynamic_color]
            r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

            # Find the closest color name using float to avoid overflow
            minimum = float('inf')
            color_name = ''
            for i in range(len(self.df)):
                dr = float(r - int(self.df.loc[i, 'R']))
                dg = float(g - int(self.df.loc[i, 'G']))
                db = float(b - int(self.df.loc[i, 'B']))
                d = abs(dr) + abs(dg) + abs(db)
                if d < minimum:
                    minimum = d
                    color_name = self.df.loc[i, 'color_name']

            # Calculate text color safely
            brightness = int(r) + int(g) + int(b)
            text_color = 'white' if brightness < 382 else 'black'

            # Update the stylesheet to apply the background color to the entire container
            self.dynamic_color_label.setStyleSheet(f"""
                QLabel {{
                    background-color: rgb({r},{g},{b});
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    margin-top: 10px;
                }}
            """)

            # Create dynamic color info text
            color_info = f"""
            <div style='color: {text_color};'>
                <h4 style='margin: 0;'>Dynamic Color</h4>
                <p style='margin: 2px 0;'>RGB: ({r}, {g}, {b})</p>
                <p style='margin: 2px 0;'>Name: {color_name}</p>
                <p style='margin: 2px 0;'>Hex: #{format(r, '02x')}{format(g, '02x')}{format(b, '02x')}</p>
            </div>
            """
            self.dynamic_color_label.setText(color_info)

    def get_color(self, event):
        if self.current_frame is not None:
            x = event.pos().x()
            y = event.pos().y()

            if 0 <= x < self.current_frame.shape[1] and 0 <= y < self.current_frame.shape[0]:
                self.current_color = self.current_frame[y, x][::-1]
                self.update_color_info()
                self.generate_and_display_combinations()

    def update_color_info(self):
        r, g, b = [int(x) for x in self.current_color]
        r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))

        # Find the closest color name using float to avoid overflow
        minimum = float('inf')
        color_name = ''
        for i in range(len(self.df)):
            dr = float(r - int(self.df.loc[i, 'R']))
            dg = float(g - int(self.df.loc[i, 'G']))
            db = float(b - int(self.df.loc[i, 'B']))
            d = abs(dr) + abs(dg) + abs(db)
            if d < minimum:
                minimum = d
                color_name = self.df.loc[i, 'color_name']

        # Calculate text color safely
        brightness = int(r) + int(g) + int(b)
        text_color = 'white' if brightness < 382 else 'black'

        # Update the stylesheet to apply the background color to the entire container
        self.color_info_label.setStyleSheet(f"""
            QLabel {{
                background-color: rgb({r},{g},{b});
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }}
        """)

        # Create color info text
        color_info = f"""
        <div style='color: {text_color};'>
            <h3 style='margin: 0;'>Selected Color</h3>
            <p style='margin: 5px 0;'>RGB: ({r}, {g}, {b})</p>
            <p style='margin: 5px 0;'>Name: {color_name}</p>
            <p style='margin: 5px 0;'>Hex: #{format(r, '02x')}{format(g, '02x')}{format(b, '02x')}</p>
        </div>
        """
        self.color_info_label.setText(color_info)

    def generate_and_display_combinations(self):
        r, g, b = self.current_color
        combinations = generate_color_combinations(r, g, b)
        combinations_image, self.color_data = create_combinations_display(combinations)

        if combinations_image is not None:
            height, width, channel = combinations_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(combinations_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.combinations_label.setPixmap(QPixmap.fromImage(q_img))

            # Clear existing buttons
            for i in reversed(range(self.copy_buttons_layout.count())):
                self.copy_buttons_layout.itemAt(i).widget().setParent(None)

            # Add new copy buttons with styling
            for combo_type in combinations.keys():
                copy_btn = QPushButton(f"Copy {combo_type.replace('_', ' ').title()} Colors")
                copy_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        padding: 5px;
                        border: none;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                copy_btn.clicked.connect(lambda checked, ct=combo_type: self.copy_color_codes(ct))
                self.copy_buttons_layout.addWidget(copy_btn)

    def change_mode(self, checked):
        self.is_video_mode = not checked
        self.capture_btn.setEnabled(self.is_video_mode)
        if self.cap is not None and not self.is_video_mode:
            self.cap.release()
            self.cap = None
            self.pause_btn.setEnabled(False)
            self.timer.stop()

    def open_file(self):
        if self.is_video_mode:
            file_filter = "Video files (*.mp4 *.avi *.mov);;All files (*.*)"
        else:
            file_filter = "Image files (*.jpg *.jpeg *.png *.bmp);;All files (*.*)"

        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)

        if file_name:
            if self.is_video_mode:
                self.cap = cv2.VideoCapture(file_name)
                self.pause_btn.setEnabled(True)
                self.timer.start()
            else:
                self.current_frame = cv2.imread(file_name)
                self.original_frame = self.current_frame.copy()
                self.update_display()
                self.update_color_palette()

    def open_webcam(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.pause_btn.setEnabled(True)
            self.is_video_mode = True
            self.video_mode_btn.setChecked(True)
            self.timer.start()
        else:
            self.statusBar.showMessage("Error: Could not open webcam", 3000)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        self.pause_btn.setText("Resume" if self.is_paused else "Pause")

    def capture_frame(self):
        if self.current_frame is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Frame", "",
                                                       "Image files (*.jpg *.jpeg *.png *.bmp);;All files (*.*)")
            if file_name:
                cv2.imwrite(file_name, self.current_frame)
                self.statusBar.showMessage(f"Frame saved to {file_name}", 3000)

    def update_frame(self):
        if self.cap is not None and not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.original_frame = frame.copy()
                self.update_display()

                current_time = time.time() * 1000
                if current_time - self.last_palette_update > self.palette_update_interval:
                    self.update_color_palette()
                    self.last_palette_update = current_time
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update_display(self):
        if self.current_frame is not None:
            display_frame = self.current_frame.copy()

            zoomed = self.create_zoomed_cursor(display_frame, self.mouse_pos[0], self.mouse_pos[1])
            if zoomed is not None:
                x, y = self.mouse_pos
                zoom_size = self.zoom_radius * 2

                zoom_x = x + 20
                zoom_y = y + 20

                if zoom_x + zoom_size * 2 > display_frame.shape[1]:
                    zoom_x = x - zoom_size * 2 - 20
                if zoom_y + zoom_size * 2 > display_frame.shape[0]:
                    zoom_y = y - zoom_size * 2 - 20

                roi = display_frame[zoom_y:zoom_y + zoom_size * 2,
                      zoom_x:zoom_x + zoom_size * 2]
                if roi.shape == zoomed.shape:
                    display_frame[zoom_y:zoom_y + zoom_size * 2,
                    zoom_x:zoom_x + zoom_size * 2] = zoomed

            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.display_label.setPixmap(QPixmap.fromImage(qt_image))

    def update_color_palette(self):
        if self.current_frame is not None:
            dominant_colors = get_dominant_colors(self.current_frame)

            palette_height = 50
            palette_width = 400
            palette_image = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)

            rect_width = palette_width // len(dominant_colors)
            for i, color in enumerate(dominant_colors):
                start_x = i * rect_width
                end_x = start_x + rect_width
                cv2.rectangle(palette_image, (start_x, 0), (end_x, palette_height),
                              (int(color[2]), int(color[1]), int(color[0])), -1)

            h, w, ch = palette_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(palette_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.palette_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ColorDetectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()