"""
Live Fire and Smoke Detection using Webcam with PPE Compliance Checking
This script uses two models:
- best.pt for fire/smoke detection
- model2.pt for PPE compliance checking
Sends email alerts when fire is detected or when non-compliant persons are found.
"""
from datetime import datetime
import sys
from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path
import json
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from datetime import datetime
import time
from dotenv import load_dotenv
import numpy as np
from collections import defaultdict
import csv

# ================= DEMO EXPIRY CONTROL =================

# DEMO_EXPIRY_DATE = datetime(2026, 3, 4, 23, 59, 59)  # Change this date

current_time = datetime.now()

# if current_time > DEMO_EXPIRY_DATE:
#     print("====================================")
#     print("        DATTU AI DEMO EXPIRED       ")
#     print(" Please contact provider to renew.  ")
#     print("====================================")
#     sys.exit()

# ========================================================

try:
    from openpyxl import Workbook, load_workbook
    OPENPYXL_AVAILABLE = True
except Exception:
    OPENPYXL_AVAILABLE = False

# ============================================================================
# CONFIGURATION - PPE Compliance Settings
# ============================================================================

# Mandatory PPE items that each person must have
# Available options: 'Boots', 'Ear-protection', 'Glass', 'Glove', 'Helmet', 'Mask', 'Vest'
MANDATORY_PPE = {'helmet', 'vest'}  # Change this to configure mandatory items

# PPE detection confidence threshold
PPE_CONFIDENCE_THRESHOLD = 0.30

# Observation period for PPE compliance (seconds)
# Person must be observed for this duration before flagging as non-compliant
PPE_OBSERVATION_PERIOD = 1.0  # 5 seconds observation window

# ============================================================================
# End of Configuration
# ============================================================================

# ===== PATH CONFIGURATION =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.join(BASE_DIR, "config")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DEFAULT_FIRE_MODEL_PATH = os.path.join(MODELS_DIR, "besttt.pt")
DEFAULT_PPE_MODEL_PATH = os.path.join(MODELS_DIR, "besttt.pt")

# Explicitly create logs and models directories for runtime.
os.makedirs("logs", exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== OBSERVATION LOGGING =====
CSV_LOG_PATH = Path(os.path.join(LOGS_DIR, "observations.csv"))
XLSX_LOG_PATH = Path(os.path.join(LOGS_DIR, "observations.xlsx"))
CSV_LOG_LOCK = threading.Lock()
XLSX_LOCK_WARNING_INTERVAL_SECONDS = 60
_last_xlsx_lock_warning_time = 0.0
OBSERVATION_HEADERS = [
    "ObservationID",
    "ObservationDate",
    "Location",
    "Category",
    "Description",
    "RiskLevel",
    "IsNearMiss",
    "IsIncident",
]
RISK_MAP = {
    "Fire Safety": "High",
    "Restricted Zone": "High",
    "Machine Safety": "High",
    "PPE": "Medium",
    "Housekeeping": "Low",
}
CAMERA_NAME_BY_ID = {}


def _ensure_csv_headers(log_path=CSV_LOG_PATH):
    """Create CSV file with headers if it does not exist."""
    if log_path.exists():
        return
    try:
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(OBSERVATION_HEADERS)
    except OSError as exc:
        print(f"[CSV Log] Failed to create log file: {exc}")


def _derive_category(event_type, message):
    event_key = str(event_type or "").strip().lower()
    message_key = str(message or "").strip().lower()
    combined = f"{event_key} {message_key}"
    if "fire" in combined:
        return "Fire Safety"
    if "restricted" in combined or "unauthorized" in combined:
        return "Restricted Zone"
    if "machine" in combined or "guard open" in combined:
        return "Machine Safety"
    if "ppe" in combined or "helmet" in combined or "mask" in combined or "vest" in combined:
        return "PPE"
    return "Housekeeping"


def _derive_description(event_type, message):
    event_key = str(event_type or "").strip().upper()
    if event_key == "FIRE":
        return "Fire detected near equipment."
    if event_key == "RESTRICTED_ZONE_OBJECT":
        return "Unauthorized entry detected."
    if event_key == "PPE_NON_COMPLIANT":
        return "Helmet missing detected."
    if event_key == "STARTUP":
        return "Monitoring session started."
    if event_key == "SHUTDOWN":
        return "Monitoring session stopped."
    clean_message = str(message or "").strip()
    if clean_message:
        return clean_message
    return "Safety observation recorded."


def _resolve_location(camera_ref):
    if isinstance(camera_ref, dict):
        return str(camera_ref.get("name", camera_ref.get("id")))
    camera_key = str(camera_ref)
    if camera_key in CAMERA_NAME_BY_ID:
        return CAMERA_NAME_BY_ID[camera_key]
    normalized_key = camera_key.strip().lower().replace("cam", "").replace("camera_", "").replace("camera", "")
    if normalized_key in CAMERA_NAME_BY_ID:
        return CAMERA_NAME_BY_ID[normalized_key]
    return camera_key


def _append_event_to_xlsx(row, log_path=XLSX_LOG_PATH):
    """Append one event row to XLSX if openpyxl is available."""
    global _last_xlsx_lock_warning_time
    if not OPENPYXL_AVAILABLE:
        return
    for _ in range(3):
        workbook = None
        try:
            if log_path.exists():
                workbook = load_workbook(log_path)
                worksheet = workbook.active
            else:
                workbook = Workbook()
                worksheet = workbook.active
                worksheet.title = "Observations"
                worksheet.append(OBSERVATION_HEADERS)
            worksheet.append(row)
            workbook.save(log_path)
            return
        except PermissionError:
            time.sleep(0.2)
        except Exception as exc:
            print(f"[XLSX Log] Failed to write event: {exc}")
            return
        finally:
            if workbook is not None:
                try:
                    workbook.close()
                except Exception:
                    pass
    now_ts = time.time()
    if now_ts - _last_xlsx_lock_warning_time >= XLSX_LOCK_WARNING_INTERVAL_SECONDS:
        print("[XLSX Log] observations.xlsx is locked by another app. Continuing with CSV logging.")
        _last_xlsx_lock_warning_time = now_ts


def append_event_to_csv(camera, event_type, message, confidence=None, snapshot_path="", log_path=CSV_LOG_PATH):
    """Append one observation row to CSV/XLSX in enterprise format."""
    try:
        with CSV_LOG_LOCK:
            _ensure_csv_headers(log_path)
            now = datetime.now()
            observation_id = "OBS_" + now.strftime("%Y%m%d_%H%M%S")
            observation_date = now.strftime("%Y-%m-%d")
            location = camera.get("name", camera.get("id")) if isinstance(camera, dict) else str(camera)
            category = _derive_category(event_type, message)
            description = _derive_description(event_type, message)
            risk_level = RISK_MAP.get(category, "Medium")
            is_near_miss = category != "Fire Safety"
            is_incident = False
            row = [
                observation_id,
                observation_date,
                location,
                category,
                description,
                risk_level,
                bool(is_near_miss),
                bool(is_incident),
            ]
            with open(log_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(row)
            _append_event_to_xlsx(row)
    except Exception as exc:
        print(f"[CSV Log] Failed to write event: {exc}")


# ===== ADDED FOR MODULE CONTROL =====
def normalize_module_flags(modules):
    """Normalize module flags with safe defaults."""
    defaults = {"fire": True, "ppe": True, "restricted_zone": True}
    if not isinstance(modules, dict):
        return defaults
    return {
        "fire": bool(modules.get("fire", True)),
        "ppe": bool(modules.get("ppe", True)),
        "restricted_zone": bool(modules.get("restricted_zone", True))
    }


# ===== ADDED FOR MULTI-CAMERA SUPPORT =====
def load_camera_config(config_path):
    """Load camera definitions from configured camera JSON file."""
    cfg_path = Path(config_path) if Path(config_path).is_absolute() else Path(BASE_DIR) / config_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Camera config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    cameras = raw.get("cameras", [])
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Camera config must contain a non-empty 'cameras' list")

    global CAMERA_NAME_BY_ID
    normalized = []
    CAMERA_NAME_BY_ID = {}
    for idx, cam in enumerate(cameras, start=1):
        if not isinstance(cam, dict):
            raise ValueError(f"Camera entry at index {idx - 1} must be an object")
        source = cam.get("source", cam.get("rtsp"))
        if source is None:
            raise ValueError(f"Camera entry at index {idx - 1} is missing 'source' or 'rtsp'")
        camera_id = str(cam.get("id", f"CAM_{idx}"))
        camera_name = str(cam.get("name", camera_id))
        CAMERA_NAME_BY_ID[camera_id] = camera_name
        CAMERA_NAME_BY_ID[str(idx)] = camera_name
        normalized.append({
            "id": camera_id,
            "name": camera_name,
            "source": source,
            "modules": normalize_module_flags(cam.get("modules", {}))
        })
    return normalized


def process_camera(camera_entry, shared_args, shared_models):
    """Process one camera stream and auto-reconnect on failure."""
    modules = camera_entry["modules"]
    camera_name = camera_entry.get("name") or camera_entry.get("id", "CAM")
    while True:
        try:
            detect_live(
                model_path=shared_args["model_path"],
                camera_index=camera_entry["source"],
                conf_threshold=shared_args["conf_threshold"],
                show_labels=shared_args["show_labels"],
                show_conf=shared_args["show_conf"],
                email_cooldown=shared_args["email_cooldown"],
                camera_name=camera_name,
                camera_context=camera_entry,
                enable_fire_module=modules["fire"],
                enable_ppe_module=modules["ppe"],
                enable_restricted_zone_module=modules["restricted_zone"],
                preloaded_fire_model=shared_models["fire_model"],
                preloaded_ppe_model=shared_models["ppe_model"]
            )
            print(f"[{camera_name}] Stream ended. Reconnecting in 5 seconds...")
        except Exception as exc:
            print(f"[{camera_name}] Stream error: {exc}. Reconnecting in 5 seconds...")
        time.sleep(5)


def load_global_models(model_path, enable_ppe=True):
    """Load YOLO models once and share across camera threads."""
    print(f"Loading fire detection model once from: {model_path}")
    fire_model = YOLO(str(model_path))
    ppe_model = None
    if enable_ppe:
        ppe_model_path = Path(DEFAULT_PPE_MODEL_PATH)
        if ppe_model_path.exists():
            print(f"Loading PPE detection model once from: {ppe_model_path}")
            ppe_model = YOLO(str(ppe_model_path))
        else:
            print("Warning: PPE model (models/besttt.pt) not found. PPE compliance checking will be disabled.")
    return {"fire_model": fire_model, "ppe_model": ppe_model}


def run_multi_camera_from_config(config_path, shared_args, shared_models):
    """Launch one thread per camera from config."""
    cameras = load_camera_config(config_path)
    threads = []
    for camera_entry in cameras:
        thread_name = str(camera_entry.get("name") or camera_entry.get("id", "CAM"))
        t = threading.Thread(
            target=process_camera,
            args=(camera_entry, shared_args, shared_models),
            daemon=False,
            name=f"cam-{thread_name}"
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

def load_email_config():
    """Load email configuration from .env file."""
    load_dotenv()
    
    config = {
        'smtp_host': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', 587)),
        'smtp_user': os.getenv('SMTP_USER', ''),
        'smtp_password': os.getenv('SMTP_PASSWORD', '').replace(' ', ''),
        'sender_email': os.getenv('SENDER_EMAIL', ''),
        'receiver_email': os.getenv('RECIEVER_EMAIL', '')
    }
    
    # Validate configuration
    if not config['smtp_user'] or not config['smtp_password']:
        raise ValueError("SMTP_USER and SMTP_PASSWORD must be set in .env file")
    if not config['receiver_email']:
        raise ValueError("RECIEVER_EMAIL must be set in .env file")
    
    return config


def normalize_camera_source(camera_source):
    """Normalize camera input from CLI into int index or URL/path string."""
    if isinstance(camera_source, int):
        return camera_source

    source = str(camera_source).strip()

    # Remove surrounding quotes if user passed quoted value.
    if (source.startswith('"') and source.endswith('"')) or (source.startswith("'") and source.endswith("'")):
        source = source[1:-1].strip()

    if source.isdigit():
        return int(source)

    return source


def open_camera_with_fallbacks(camera_source):
    """Open camera with backend fallbacks for webcam and RTSP sources."""
    source = normalize_camera_source(camera_source)
    attempts = []

    if isinstance(source, int):
        for backend_name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
            if hasattr(cv2, backend_name):
                attempts.append((backend_name, getattr(cv2, backend_name)))
    else:
        source_lower = source.lower()
        if source_lower.startswith("rtsp://"):
            # Improve RTSP stability on some builds/network conditions.
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

            for backend_name in ("CAP_FFMPEG", "CAP_GSTREAMER", "CAP_ANY"):
                if hasattr(cv2, backend_name):
                    attempts.append((backend_name, getattr(cv2, backend_name)))
        else:
            for backend_name in ("CAP_FFMPEG", "CAP_ANY"):
                if hasattr(cv2, backend_name):
                    attempts.append((backend_name, getattr(cv2, backend_name)))

    errors = []
    for backend_name, backend_value in attempts:
        try:
            cap = cv2.VideoCapture(source, backend_value)
        except Exception as exc:
            errors.append(f"{backend_name}: exception: {exc}")
            continue

        if cap.isOpened():
            return cap, source, backend_name

        cap.release()
        errors.append(f"{backend_name}: not opened")

    # Final fallback with default backend selection.
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap, source, "DEFAULT"
    cap.release()

    details = "; ".join(errors) if errors else "No backend attempts were available"
    raise RuntimeError(
        f"Failed to open camera source '{source}'. Tried backends -> {details}"
    )


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) tuple
        polygon: List of (x, y) tuples representing polygon vertices
    
    Returns:
        True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def box_intersects_polygon(box, polygon):
    """
    Check if a bounding box intersects with a polygon.
    
    Args:
        box: Bounding box coordinates [x1, y1, x2, y2]
        polygon: List of (x, y) tuples representing polygon vertices
    
    Returns:
        True if box intersects polygon, False otherwise
    """
    x1, y1, x2, y2 = box
    
    # Check if any corner of the box is inside the polygon
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for corner in corners:
        if point_in_polygon(corner, polygon):
            return True
    
    # Check if center of box is inside polygon
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    if point_in_polygon(center, polygon):
        return True
    
    return False


def draw_quadrilateral_interactive(frame):
    """
    Interactive function to draw a quadrilateral by clicking 4 points.
    
    Args:
        frame: Input frame to draw on
    
    Returns:
        List of 4 (x, y) tuples representing the quadrilateral corners, or None if cancelled
    """
    points = []
    temp_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_frame
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
                
                if len(points) > 1:
                    # Draw line to previous point
                    cv2.line(temp_frame, points[-2], points[-1], (0, 255, 0), 2)
                
                if len(points) == 4:
                    # Close the quadrilateral
                    cv2.line(temp_frame, points[3], points[0], (0, 255, 0), 2)
                    # Fill with semi-transparent overlay
                    pts = np.array(points, np.int32)
                    overlay = temp_frame.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    cv2.addWeighted(overlay, 0.3, temp_frame, 0.7, 0, temp_frame)
    
    cv2.namedWindow('Draw Restricted Zone - Click 4 points', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Draw Restricted Zone - Click 4 points', mouse_callback)
    
    instructions = [
        "Click 4 points to define the restricted zone",
        "Points will be connected in order",
        "Press 'Enter' to confirm, 'Esc' to cancel"
    ]
    
    while len(points) < 4:
        display_frame = temp_frame.copy()
        
        # Draw instructions
        y_offset = 30
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show point count
        cv2.putText(display_frame, f"Points selected: {len(points)}/4", (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Draw Restricted Zone - Click 4 points', display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            cv2.destroyWindow('Draw Restricted Zone - Click 4 points')
            return None
        elif key == 13:  # Enter
            if len(points) == 4:
                break
    
    cv2.destroyWindow('Draw Restricted Zone - Click 4 points')
    return points if len(points) == 4 else None


def send_unidentified_object_email(email_config, image_path, objects_info):
    """
    Send email alert when unidentified objects are detected in restricted zone.
    
    Args:
        email_config: Email configuration dictionary
        image_path: Path to the snapshot image
        objects_info: List of dicts with object info (class_name, confidence, bbox)
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email'] or email_config['smtp_user']
        msg['To'] = email_config['receiver_email']
        msg['Subject'] = f"⚠️ UNIDENTIFIED OBJECT DETECTED - Alert at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        timestamp = datetime.now()
        
        # Create object details list
        objects_html = ""
        for i, obj in enumerate(objects_info, 1):
            objects_html += f"""
            <li>
                <strong>Object {i}:</strong> {obj['class_name']}<br>
                &nbsp;&nbsp;Confidence: {obj['confidence']:.2%}<br>
                &nbsp;&nbsp;Location: ({obj['bbox'][0]:.0f}, {obj['bbox'][1]:.0f}) to ({obj['bbox'][2]:.0f}, {obj['bbox'][3]:.0f})
            </li>
            """
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #ff9800;">⚠️ UNIDENTIFIED OBJECT DETECTION ALERT</h2>
            
            <div style="background-color: #fff3e0; padding: 15px; border-left: 4px solid #ff9800; margin: 20px 0;">
                <h3 style="margin-top: 0; color: #f57c00;">Non-person object(s) detected in restricted zone!</h3>
                <p>An object other than a person has been detected in the monitored restricted area for more than 10 seconds.</p>
            </div>
            
            <h3>Detection Details:</h3>
            <ul>
                <li><strong>Date:</strong> {timestamp.strftime('%A, %B %d, %Y')}</li>
                <li><strong>Time:</strong> {timestamp.strftime('%I:%M:%S %p')}</li>
                <li><strong>Timestamp:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>
                <li><strong>Timezone:</strong> {time.tzname[0] if time.tzname else 'Local Time'}</li>
                <li><strong>Number of Objects:</strong> {len(objects_info)}</li>
            </ul>
            
            <h3>Detected Objects:</h3>
            <ul>
                {objects_html}
            </ul>
            
            <h3>Action Required:</h3>
            <p>Please review the attached image with bounding boxes and investigate the unidentified object(s) in the restricted zone.</p>
            
            <hr style="border: 1px solid #ddd; margin: 20px 0;">
            
            <p style="color: #666; font-size: 12px;">
                This is an automated alert from the Restricted Zone Monitoring System.<br>
                Generated at: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach image
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data)
            image.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(image)
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['smtp_user'], email_config['smtp_password'])
        server.send_message(msg)
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def calculate_bbox_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_bbox_center(bbox):
    """Calculate center point of a bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) / 2, (y_min + y_max) / 2)

def calculate_distance(center1, center2):
    """Calculate Euclidean distance between two center points."""
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def is_ppe_near_person(ppe_bbox, person_bbox, overlap_threshold=0.1, distance_threshold=200):
    """
    Check if a PPE item is associated with a person.
    Uses both overlap and proximity checks.
    """
    # Check overlap
    overlap = calculate_bbox_overlap(ppe_bbox, person_bbox)
    if overlap > overlap_threshold:
        return True
    
    # Check proximity (if PPE is within reasonable distance of person)
    ppe_center = calculate_bbox_center(ppe_bbox)
    person_center = calculate_bbox_center(person_bbox)
    distance = calculate_distance(ppe_center, person_center)
    
    # Also check if PPE center is inside person bbox (with some margin)
    x_min, y_min, x_max, y_max = person_bbox
    margin = 50  # pixels
    if (x_min - margin <= ppe_center[0] <= x_max + margin and 
        y_min - margin <= ppe_center[1] <= y_max + margin):
        return True
    
    # Check if distance is within threshold
    if distance < distance_threshold:
        return True
    
    return False

def check_ppe_compliance(persons, ppe_items, mandatory_ppe, class_names, person_tracking, current_time):
    """
    Check PPE compliance for each person with 5-second observation window.
    Tracks if mandatory PPE items are detected AT LEAST ONCE during observation period.
    
    Args:
        persons: List of person detections [(bbox, confidence), ...]
        ppe_items: Dict of PPE items by class {class_name: [(bbox, confidence), ...]}
        mandatory_ppe: Set of mandatory PPE class names
        class_names: Dict mapping class_id to class_name
        person_tracking: Dict tracking persons over time {person_id: {'first_seen': time, 'detected_ppe': set, 'last_bbox': bbox}}
        current_time: Current timestamp
    
    Returns:
        List of compliance results for each person
    """
    compliance_results = []
    
    for person_idx, (person_bbox, person_conf) in enumerate(persons):
        # Find all PPE items associated with this person in current frame
        current_frame_ppe = set()
        
        for ppe_class, items in ppe_items.items():
            if ppe_class == 'Person':
                continue
            
            for ppe_bbox, ppe_conf in items:
                if is_ppe_near_person(ppe_bbox, person_bbox):
                    current_frame_ppe.add(ppe_class)
        
        # Try to match this person with existing tracked person
        matched_person_id = None
        best_overlap = 0.3
        
        for person_id, track_data in person_tracking.items():
            last_bbox = track_data.get('last_bbox')
            if last_bbox:
                overlap = calculate_bbox_overlap(person_bbox, last_bbox)
                if overlap > best_overlap:
                    matched_person_id = person_id
                    best_overlap = overlap
        
        if matched_person_id is not None:
            # Update existing tracked person
            person_id = matched_person_id
            person_tracking[person_id]['detected_ppe'].update(current_frame_ppe)  # Add any new PPE detections
            person_tracking[person_id]['last_bbox'] = person_bbox
            person_tracking[person_id]['last_seen'] = current_time
        else:
            # New person detected - start tracking
            person_id = f"person_{len(person_tracking)}"
            person_tracking[person_id] = {
                'first_seen': current_time,
                'detected_ppe': current_frame_ppe.copy(),  # Track PPE items detected at least once
                'last_bbox': person_bbox,
                'last_seen': current_time
            }
        
        # Get tracking data
        track_data = person_tracking[person_id]
        time_since_first_seen = current_time - track_data['first_seen']
        detected_ppe_set = track_data['detected_ppe']
        
        # Check compliance based on observation period
        if time_since_first_seen < PPE_OBSERVATION_PERIOD:
            # Still in observation period - show as "OBSERVING"
            missing_ppe = [ppe for ppe in mandatory_ppe if ppe not in detected_ppe_set]
            present_ppe = [ppe for ppe in mandatory_ppe if ppe in detected_ppe_set]
            
            compliance_results.append({
                'person_idx': person_idx,
                'person_bbox': person_bbox,
                'person_conf': person_conf,
                'present_ppe': present_ppe,
                'missing_ppe': missing_ppe,
                'is_compliant': None,  # None = still observing
                'is_observing': True,
                'observation_time': time_since_first_seen,
                'person_ppe': detected_ppe_set
            })
        else:
            # Observation period complete - check final compliance
            missing_ppe = [ppe for ppe in mandatory_ppe if ppe not in detected_ppe_set]
            present_ppe = [ppe for ppe in mandatory_ppe if ppe in detected_ppe_set]
            
            compliance_results.append({
                'person_idx': person_idx,
                'person_bbox': person_bbox,
                'person_conf': person_conf,
                'present_ppe': present_ppe,
                'missing_ppe': missing_ppe,
                'is_compliant': len(missing_ppe) == 0,  # Compliant only if all mandatory PPE detected at least once
                'is_observing': False,
                'observation_time': time_since_first_seen,
                'person_ppe': detected_ppe_set
            })
    
    # Clean up tracking for persons no longer detected (remove if not seen for 2 seconds)
    persons_to_remove = []
    for person_id, track_data in person_tracking.items():
        if current_time - track_data.get('last_seen', current_time) > 2.0:
            persons_to_remove.append(person_id)
    
    for person_id in persons_to_remove:
        del person_tracking[person_id]
    
    return compliance_results


def send_ppe_non_compliant_email(email_config, image_path, non_compliant_persons):
    """
    Send email alert when non-compliant persons are detected.
    
    Args:
        email_config: Email configuration dictionary
        image_path: Path to the snapshot image
        non_compliant_persons: List of non-compliant person data
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email'] or email_config['smtp_user']
        msg['To'] = email_config['receiver_email']
        msg['Subject'] = f"⚠️ PPE NON-COMPLIANCE ALERT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create email body
        timestamp = datetime.now()
        num_non_compliant = len(non_compliant_persons)
        
        # Build list of missing PPE for each person
        missing_details = []
        for i, person in enumerate(non_compliant_persons, 1):
            missing_ppe = person.get('missing_ppe', [])
            missing_text = ', '.join(missing_ppe) if missing_ppe else 'Unknown'
            missing_details.append(f"<li><strong>Person {i}:</strong> Missing - {missing_text}</li>")
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #ff9800;">⚠️ PPE NON-COMPLIANCE ALERT</h2>
            
            <div style="background-color: #fff3e0; padding: 15px; border-left: 4px solid #ff9800; margin: 20px 0;">
                <h3 style="margin-top: 0; color: #f57c00;">{num_non_compliant} Non-Compliant Person(s) Detected!</h3>
            </div>
            
            <h3>Detection Details:</h3>
            <ul>
                <li><strong>Date:</strong> {timestamp.strftime('%A, %B %d, %Y')}</li>
                <li><strong>Time:</strong> {timestamp.strftime('%I:%M:%S %p')}</li>
                <li><strong>Timestamp:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>
                <li><strong>Number of Non-Compliant Persons:</strong> {num_non_compliant}</li>
            </ul>
            
            <h3>Missing PPE Items:</h3>
            <ul>
                {''.join(missing_details)}
            </ul>
            
            <h3>Mandatory PPE Required:</h3>
            <ul>
                <li>{', '.join(MANDATORY_PPE)}</li>
            </ul>
            
            <h3>Action Required:</h3>
            <p>Please review the attached image and ensure all personnel are wearing the required PPE.</p>
            
            <hr style="border: 1px solid #ddd; margin: 20px 0;">
            
            <p style="color: #666; font-size: 12px;">
                This is an automated alert from the PPE Compliance Detection System.<br>
                Generated at: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach image
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data)
            image.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(image)
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['smtp_user'], email_config['smtp_password'])
        server.send_message(msg)
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_fire_alert_email(email_config, image_path, detection_count, max_confidence):
    """
    Send email alert when fire is detected.
    
    Args:
        email_config: Email configuration dictionary
        image_path: Path to the snapshot image
        detection_count: Number of fire detections
        max_confidence: Maximum confidence score of detections
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email'] or email_config['smtp_user']
        msg['To'] = email_config['receiver_email']
        msg['Subject'] = f"🚨 FIRE DETECTED - Alert at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create email body with detailed timestamp
        timestamp = datetime.now()
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2 style="color: #d32f2f;">🚨 FIRE DETECTION ALERT</h2>
            
            <div style="background-color: #ffebee; padding: 15px; border-left: 4px solid #d32f2f; margin: 20px 0;">
                <h3 style="margin-top: 0; color: #c62828;">Fire has been detected in the monitored area!</h3>
            </div>
            
            <h3>Detection Details:</h3>
            <ul>
                <li><strong>Date:</strong> {timestamp.strftime('%A, %B %d, %Y')}</li>
                <li><strong>Time:</strong> {timestamp.strftime('%I:%M:%S %p')}</li>
                <li><strong>Timestamp:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</li>
                <li><strong>Timezone:</strong> {time.tzname[0] if time.tzname else 'Local Time'}</li>
                <li><strong>Number of Fire Detections:</strong> {detection_count}</li>
                <li><strong>Maximum Confidence:</strong> {max_confidence:.2%}</li>
            </ul>
            
            <h3>Action Required:</h3>
            <p>Please review the attached image and take appropriate action immediately.</p>
            
            <hr style="border: 1px solid #ddd; margin: 20px 0;">
            
            <p style="color: #666; font-size: 12px;">
                This is an automated alert from the Fire Detection System.<br>
                Generated at: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}
            </p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Attach image
        if os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img_data = f.read()
            image = MIMEImage(img_data)
            image.add_header('Content-Disposition', f'attachment; filename={os.path.basename(image_path)}')
            msg.attach(image)
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_host'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['smtp_user'], email_config['smtp_password'])
        server.send_message(msg)
        server.quit()
        
        return True
    
    except Exception as e:
        print(f"Error sending email: {e}")
        return False


def detect_live(
    model_path=DEFAULT_FIRE_MODEL_PATH,
    camera_index=0,
    conf_threshold=0.40,
    show_labels=True,
    show_conf=True,
    email_cooldown=60,  # Seconds between emails
    camera_name="CAM_1",
    camera_context=None,
    enable_fire_module=True,
    enable_ppe_module=True,
    enable_restricted_zone_module=True,
    preloaded_fire_model=None,
    preloaded_ppe_model=None
):
    """
    Run live detection on camera feed.
    
    Args:
        model_path: Path to the trained model (.pt file)
        camera_index: Camera device index (usually 0 for default camera)
        conf_threshold: Confidence threshold for detections (0.0 to 1.0)
        show_labels: Whether to show class labels
        show_conf: Whether to show confidence scores
        email_cooldown: Minimum seconds between email alerts
    """
    
    # Load email configuration
    try:
        email_config = load_email_config()
        print("Email configuration loaded successfully")
    except Exception as e:
        print(f"Warning: Email configuration error: {e}")
        print("Email alerts will be disabled")
        email_config = None
    
    # Create detect directory for snapshots
    detect_dir = Path('detect')
    detect_dir.mkdir(exist_ok=True)
    print(f"Snapshots will be saved to: {detect_dir.absolute()}")
    
    # Load fire/PPE-item detection model (besttt.pt)
    if preloaded_fire_model is not None:
        fire_model = preloaded_fire_model
    else:
        model_path = Path(DEFAULT_FIRE_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Place your fire-detection YOLO weights file as 'besttt.pt' in the 'models' folder, "
                "or set the correct path when calling detect_live()."
            )
        print(f"Loading fire detection model from: {model_path}")
        fire_model = YOLO(str(model_path))
    
    # ===== ADDED FOR MODULE CONTROL =====
    ppe_model = None
    if enable_ppe_module:
        if preloaded_ppe_model is not None:
            ppe_model = preloaded_ppe_model
        else:
            # Load PPE detection model (models/besttt
            # .pt)
            ppe_model_path = Path(DEFAULT_PPE_MODEL_PATH)
            if not ppe_model_path.exists():
                print("Warning: PPE model (models/besttt.pt) not found. PPE compliance checking will be disabled.")
                ppe_model = None
            else:
                print(f"Loading PPE detection model from: {ppe_model_path}")
                ppe_model = YOLO(str(ppe_model_path))
    else:
        print("PPE module disabled for this camera.")
    
    # Get class names from PPE model
    ppe_class_names = ppe_model.names if ppe_model and hasattr(ppe_model, 'names') else {}
    if ppe_model:
        print(f"PPE Model classes: {ppe_class_names}")
        print(f"Mandatory PPE items: {MANDATORY_PPE}")
    
    # Open camera
    print(f"Opening camera {camera_index} [{camera_name}]...")
    cap, resolved_camera_source, backend_used = open_camera_with_fallbacks(camera_index)
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera opened successfully!")
    print(f"Resolved camera source: {resolved_camera_source}")
    print(f"OpenCV backend used: {backend_used}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps if fps > 0 else 'Unknown'}")
    
    # ===== ADDED FOR MODULE CONTROL =====
    if enable_restricted_zone_module:
        # Get initial frame for quadrilateral selection
        print("\n" + "="*50)
        print("SETUP: Draw Restricted Zone")
        print("="*50)
        ret, setup_frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to read initial frame from camera")
        
        # Draw quadrilateral interactively
        restricted_zone = draw_quadrilateral_interactive(setup_frame)
        
        if restricted_zone is None:
            print("Restricted zone selection cancelled. Exiting...")
            cap.release()
            return
    else:
        print("Restricted zone module disabled for this camera.")
        restricted_zone = [
            (0, 0),
            (max(width - 1, 0), 0),
            (max(width - 1, 0), max(height - 1, 0)),
            (0, max(height - 1, 0))
        ]
    
    if enable_restricted_zone_module:
        print(f"\n✓ Restricted zone defined with 4 points:")
        for i, point in enumerate(restricted_zone, 1):
            print(f"   Point {i}: {point}")

    log_camera = camera_context if isinstance(camera_context, dict) else {"id": str(camera_name), "name": str(camera_name)}
    #     if isinstance(camera_context, dict):
    #      log_camera = camera_context
    #     else:
    #      resolved_name = CAMERA_NAME_BY_ID.get(str(camera_context), str(camera_context))
    #      log_camera = {
    #     "id": str(camera_context),
    #     "name": resolved_name
    # }
#     location_name = _resolve_location(camera_context)
#     log_camera = {
#         "id": str(camera_context) if not isinstance(camera_context, dict) else camera_context.get("id"),
#         "name": location_name
# }



    # ===== CSV LOGGING =====
    append_event_to_csv(
        camera=log_camera,
        event_type="STARTUP",
        message=(
            f"source={resolved_camera_source}, fire={enable_fire_module}, "
            f"ppe={enable_ppe_module}, restricted_zone={enable_restricted_zone_module}"
        )
    )
    print("\n" + "="*50)
    print("Live Detection Started!")
    print("Press 'q' to quit")
    print("Press 's' to save current frame")
    if enable_restricted_zone_module:
        print("Press 'r' to redraw restricted zone")
    if email_config:
        if ppe_model:
            print("Email alerts: ENABLED (fire + unidentified objects + PPE non-compliance)")
        else:
            print("Email alerts: ENABLED (fire + unidentified objects)")
    else:
        print("Email alerts: DISABLED")
    print("="*50 + "\n")
    
    frame_count = 0
    save_count = 0
    last_email_time = 0  # Track last email send time
    last_unidentified_email_time = 0  # Track last unidentified object email
    last_ppe_email_time = 0  # Track last PPE non-compliance email
    fire_detection_count = 0  # Total fire detections
    ppe_non_compliant_count = 0  # Total non-compliant instances
    
    # Track persons for PPE compliance (with 5-second observation window)
    # Format: {person_id: {'first_seen': time, 'detected_ppe': set, 'last_bbox': bbox, 'last_seen': time}}
    person_ppe_tracking = {}
    
    # Track objects in restricted zone: {object_id: {'first_seen': time, 'class_name': str, 'bbox': [x1,y1,x2,y2], 'confidence': float}}
    objects_in_zone = {}  # Track current frame objects
    persistent_objects = {}  # Track objects that have been in zone for >10s
    object_id_counter = 0
    alert_threshold = 10.0  # Seconds
    
    # Get class names from fire model
    fire_class_names = fire_model.names if hasattr(fire_model, 'names') else {}
    person_class_id = None
    # Find "Person" class ID in fire model (case-insensitive)
    for class_id, class_name in fire_class_names.items():
        if 'person' in str(class_name).lower():
            person_class_id = class_id
            break
    
    if enable_restricted_zone_module:
        if person_class_id is None:
            print("Warning: 'Person' class not found in fire model. All objects will be considered unidentified.")
        else:
            print(f"Person class ID: {person_class_id} (allowed in restricted zone)")
    
    # Store the restricted zone as background
    restricted_zone_background = None
    background_initialized = False
    background_init_frames = 30  # Frames to capture stable background
    background_frames = 0
    
    # Object detection parameters
    min_object_area = 500   # Minimum area in pixels
    min_object_width = 20   # Minimum width in pixels
    min_object_height = 20  # Minimum height in pixels
    change_threshold = 30    # Threshold for detecting changes (0-255)
    
    if enable_restricted_zone_module:
        print(f"Detection parameters: min_area={min_object_area}, min_size={min_object_width}x{min_object_height}")
        print("Background will be captured from the restricted zone area")
    
    # Create and set window to fullscreen
    window_name = f'Fire and Smoke Detection - {camera_name}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to read frame from camera")
                break
            
            # Get current time for tracking
            current_time = time.time()
            
            # Run fire detection inference
            fire_results = fire_model(frame, conf=conf_threshold, verbose=False)

            # Extract PERSON from besttt.pt (new model)
            persons = []

            if ppe_model is not None:
                person_results = ppe_model(frame, conf=0.40, verbose=False)

                if person_results[0].boxes is not None:
                    for box in person_results[0].boxes:
                        cls_id = int(box.cls[0])
                        class_name = ppe_model.names.get(cls_id, "")

                        if "person" in class_name.lower():
                            confidence = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                            persons.append((bbox, confidence))

            ppe_items = defaultdict(list)
            compliance_results = []

            if fire_results[0].boxes is not None:
                for box in fire_results[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = fire_model.names.get(cls_id, "")

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [float(x1), float(y1), float(x2), float(y2)]

                    if class_name.lower() in ["helmet", "vest"]:
                        ppe_items[class_name].append((bbox, confidence))

            # Check PPE compliance with 5-second observation window
            if ppe_model and len(persons) > 0:
                compliance_results = check_ppe_compliance(
                    persons, ppe_items, MANDATORY_PPE, ppe_class_names,
                    person_ppe_tracking, current_time
                )

                # Count non-compliant persons (only after observation period)
                for result in compliance_results:
                    if result.get('is_compliant') is False:  # Explicitly False (not None/observing)
                        ppe_non_compliant_count += 1
            
            # Check for fire detections (class 0 = fire, class 1 = smoke)
            fire_detected = False
            fire_count = 0
            max_fire_confidence = 0.0
            
            # Track objects in restricted zone
            current_frame_objects = {}  # Reset for this frame
            
            # Extract restricted zone region from frame
            pts = np.array(restricted_zone, np.int32)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            
            # Get bounding rectangle of restricted zone
            x_min = int(min(p[0] for p in restricted_zone))
            y_min = int(min(p[1] for p in restricted_zone))
            x_max = int(max(p[0] for p in restricted_zone))
            y_max = int(max(p[1] for p in restricted_zone))
            
            # Extract the restricted zone region
            zone_region = frame[y_min:y_max, x_min:x_max].copy()
            zone_mask_region = mask[y_min:y_max, x_min:x_max]
            
            # Initialize background from restricted zone
            if not background_initialized:
                # Capture background from the restricted zone area
                restricted_zone_background = zone_region.copy()
                background_frames += 1
                if background_frames >= background_init_frames:
                    background_initialized = True
                    print(f"✓ Background captured from restricted zone ({x_max-x_min}x{y_max-y_min} pixels)")
                    print("Now monitoring for changes in the restricted zone...")
            
            # Detect changes in restricted zone (only after background is set)
            change_mask = None  # Initialize for use later
            if background_initialized:
                # Compare current zone region with background
                diff = cv2.absdiff(zone_region, restricted_zone_background)
                
                # Convert to grayscale for better comparison
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                # Apply threshold to find significant changes
                _, change_mask = cv2.threshold(gray_diff, change_threshold, 255, cv2.THRESH_BINARY)
                
                # Apply zone mask to only detect changes within the zone
                change_mask = cv2.bitwise_and(change_mask, zone_mask_region)
                
                # Clean up the mask more aggressively to reduce noise
                kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel_small)  # Remove small noise
                change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel_medium)  # Fill small holes
                
                # Apply Gaussian blur to smooth
                change_mask = cv2.GaussianBlur(change_mask, (5, 5), 0)
                _, change_mask = cv2.threshold(change_mask, 50, 255, cv2.THRESH_BINARY)
                
                # Find contours of changed areas
                contours, _ = cv2.findContours(change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Process each contour
                for contour in contours:
                    # Filter small contours
                    area = cv2.contourArea(contour)
                    if area < min_object_area:
                        continue
                    
                    # Get tighter bounding box using contour approximation
                    # First, approximate the contour to reduce noise
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # Get bounding box from approximated contour (tighter fit)
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Filter by minimum size
                    if w < min_object_width or h < min_object_height:
                        continue
                    
                    # Further tighten the bounding box by finding the actual object region
                    # Extract the region of the contour
                    mask_contour = np.zeros(change_mask.shape, dtype=np.uint8)
                    cv2.drawContours(mask_contour, [contour], -1, 255, -1)
                    
                    # Find the tightest bounding box by getting non-zero pixels
                    coords = np.column_stack(np.where(mask_contour > 0))
                    if len(coords) > 0:
                        y_min_contour, x_min_contour = coords.min(axis=0)
                        y_max_contour, x_max_contour = coords.max(axis=0)
                        # Use the tighter bounding box
                        x = x_min_contour
                        y = y_min_contour
                        w = x_max_contour - x_min_contour + 1
                        h = y_max_contour - y_min_contour + 1
                    
                    # Add small padding (2 pixels) for better visibility
                    padding = 2
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(zone_region.shape[1] - x, w + 2 * padding)
                    h = min(zone_region.shape[0] - y, h + 2 * padding)
                    
                    # Convert to full frame coordinates
                    x_full = x + x_min
                    y_full = y + y_min
                    bbox = [float(x_full), float(y_full), float(x_full + w), float(y_full + h)]
                    
                    # Check if this object overlaps with any model-detected person
                    is_person = False
                    if fire_results[0].boxes is not None and len(fire_results[0].boxes) > 0:
                        for box in fire_results[0].boxes:
                            cls_id = int(box.cls[0])
                            if person_class_id is not None and cls_id == person_class_id:
                                # Check overlap with person detection
                                model_bbox = box.xyxy[0].cpu().numpy()
                                overlap = calculate_bbox_overlap(bbox, [float(model_bbox[0]), float(model_bbox[1]), 
                                                                       float(model_bbox[2]), float(model_bbox[3])])
                                if overlap > 0.5:  # 50% overlap = same object
                                    is_person = True
                                    break
                    
                    # If not a person, track it as unidentified object
                    if not is_person:
                        # Try to match with existing object
                        matched_id = None
                        best_overlap = 0.3
                        for obj_id, obj_data in objects_in_zone.items():
                            obj_bbox = obj_data['bbox']
                            overlap = calculate_bbox_overlap(bbox, obj_bbox)
                            if overlap > best_overlap:
                                matched_id = obj_id
                                best_overlap = overlap
                        
                        if matched_id is not None:
                            # Update existing object
                            current_frame_objects[matched_id] = {
                                'first_seen': objects_in_zone[matched_id]['first_seen'],
                                'class_name': 'Unidentified Object',
                                'bbox': bbox,
                                'confidence': 1.0,
                                'detection_method': 'zone_change'
                            }
                        else:
                            # New object detected
                            object_id_counter += 1
                            current_frame_objects[object_id_counter] = {
                                'first_seen': current_time,
                                'class_name': 'Unidentified Object',
                                'bbox': bbox,
                                'confidence': 1.0,
                                'detection_method': 'zone_change'
                            }
                
                # Remove objects that are no longer detected (they've been removed or moved out)
                # Objects detected via zone_change method should be removed if not found in current frame
                for obj_id in list(objects_in_zone.keys()):
                    obj_data = objects_in_zone[obj_id]
                    # Only remove zone_change detections immediately (not model detections, they might be temporarily missed)
                    if obj_data.get('detection_method') == 'zone_change' and obj_id not in current_frame_objects:
                        # Object disappeared - remove it
                        pass  # Will be removed when we update objects_in_zone = current_frame_objects
            
            # Also check fire model detections for non-person objects
            if fire_results[0].boxes is not None and len(fire_results[0].boxes) > 0:
                for box in fire_results[0].boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [float(x1), float(y1), float(x2), float(y2)]
                    
                    # Filter small detections (relaxed for model detections)
                    w = x2 - x1
                    h = y2 - y1
                    # Model detections are more reliable, so use smaller minimum size
                    if w < min_object_width * 0.7 or h < min_object_height * 0.7:
                        continue
                    
                    # Check if detection is fire (class 0) - but only if confidence is high enough
                    if enable_fire_module and cls_id == 0:  # Class 0 = fire
                        # Higher threshold for fire to avoid false positives on light sources
                        if confidence >= 0.75:  # Require 60% confidence for fire
                            fire_detected = True
                            fire_count += 1
                            if confidence > max_fire_confidence:
                                max_fire_confidence = confidence
                    
                    # Check if box intersects with restricted zone
                    if box_intersects_polygon(bbox, restricted_zone):
                        # Check if it's a person (allowed) or other object
                        if person_class_id is None or cls_id != person_class_id:
                            # Non-person object in restricted zone
                            class_name = fire_class_names.get(cls_id, f"Unidentified Object (Class_{cls_id})")
                            
                            # Filter low confidence detections
                            if confidence < 0.5:  # Ignore very low confidence detections
                                continue
                            
                            # Try to match with existing object (simple matching by overlap)
                            matched_id = None
                            best_overlap = 0.3
                            for obj_id, obj_data in objects_in_zone.items():
                                obj_bbox = obj_data['bbox']
                                overlap = calculate_bbox_overlap(bbox, obj_bbox)
                                if overlap > best_overlap:  # Find best match
                                    matched_id = obj_id
                                    best_overlap = overlap
                            
                            if matched_id is not None:
                                # Update existing object with model detection info
                                current_frame_objects[matched_id] = {
                                    'first_seen': objects_in_zone[matched_id]['first_seen'],
                                    'class_name': class_name,  # Update with model class name
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'detection_method': 'model'
                                }
                            else:
                                # New object from model - add immediately (model detections are more reliable)
                                object_id_counter += 1
                                current_frame_objects[object_id_counter] = {
                                    'first_seen': current_time,
                                    'class_name': class_name,
                                    'bbox': bbox,
                                    'confidence': confidence,
                                    'detection_method': 'model'
                                }
            
            # Update objects_in_zone
            # Objects not in current_frame_objects are automatically removed (they disappeared)
            # For zone_change objects, verify they still have significant change in their region
            if background_initialized and change_mask is not None:
                objects_to_keep = {}
                for obj_id, obj_data in current_frame_objects.items():
                    # For zone_change detections, double-check the change still exists
                    if obj_data.get('detection_method') == 'zone_change':
                        x1, y1, x2, y2 = [int(coord) for coord in obj_data['bbox']]
                        # Convert to zone region coordinates
                        x1_zone = max(0, x1 - x_min)
                        y1_zone = max(0, y1 - y_min)
                        x2_zone = min(change_mask.shape[1], x2 - x_min)
                        y2_zone = min(change_mask.shape[0], y2 - y_min)
                        
                        # Check if there's still significant change in this region
                        if x2_zone > x1_zone and y2_zone > y1_zone:
                            region_mask = change_mask[y1_zone:y2_zone, x1_zone:x2_zone]
                            change_pixels = np.sum(region_mask > 0)
                            region_area = (x2_zone - x1_zone) * (y2_zone - y1_zone)
                            if region_area > 0:
                                change_ratio = change_pixels / region_area
                                # If less than 15% of the region has changed, object is gone (background returned)
                                if change_ratio < 0.15:
                                    continue  # Skip this object - background has returned
                    
                    # Keep this object
                    objects_to_keep[obj_id] = obj_data
                
                # Update with filtered objects (removes objects where background returned)
                objects_in_zone = objects_to_keep
            else:
                # Background not initialized yet, just use current_frame_objects
                objects_in_zone = current_frame_objects
            
            # Check for objects that have been in zone for >10 seconds
            unidentified_objects = []
            for obj_id, obj_data in objects_in_zone.items():
                time_in_zone = current_time - obj_data['first_seen']
                if time_in_zone >= alert_threshold:
                    if obj_id not in persistent_objects:
                        persistent_objects[obj_id] = obj_data
                        unidentified_objects.append(obj_data)
            
            # Draw fire detection results on frame
            annotated_frame = fire_results[0].plot(
                labels=show_labels,
                conf=show_conf,
                line_width=2
            )
            
            # Draw PPE item detections on frame (from besttt.pt only)
            if ppe_model and len(ppe_items) > 0:
                for class_name, detections in ppe_items.items():
                    for bbox, confidence in detections:
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        label = f"{class_name} {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Draw PPE compliance information for each person
            if ppe_model and len(compliance_results) > 0:
                for result in compliance_results:
                    person_bbox = result['person_bbox']
                    x1, y1, x2, y2 = [int(coord) for coord in person_bbox]
                    is_observing = result.get('is_observing', False)
                    observation_time = result.get('observation_time', 0)
                    
                    if is_observing:
                        # Yellow border during observation period
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        status_text = f"OBSERVING ({observation_time:.1f}s)"
                        status_color = (0, 255, 255)  # Yellow
                        
                        # Show detected PPE so far
                        detected_ppe = result.get('person_ppe', set())
                        missing_ppe = [ppe for ppe in MANDATORY_PPE if ppe not in detected_ppe]
                        if missing_ppe:
                            info_text = f"Waiting for: {', '.join(missing_ppe)}"
                            text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                            text_x = x1
                            text_y = y1 - 10
                            
                            # Background for text
                            cv2.rectangle(annotated_frame, 
                                         (text_x - 5, text_y - text_size[1] - 5),
                                         (text_x + text_size[0] + 5, text_y + 5),
                                         (0, 0, 0), -1)
                            
                            cv2.putText(annotated_frame, info_text, (text_x, text_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    elif result['is_compliant']:
                        # Green border for compliant person
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        status_text = "COMPLIANT"
                        status_color = (0, 255, 0)
                    else:
                        # Red border for non-compliant person (after observation period)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        status_text = "NON-COMPLIANT"
                        status_color = (0, 0, 255)
                        
                        # Draw missing PPE warning
                        missing_text = f"Missing: {', '.join(result['missing_ppe'])}"
                        text_size = cv2.getTextSize(missing_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        text_x = x1
                        text_y = y1 - 10
                        
                        # Background for text
                        cv2.rectangle(annotated_frame, 
                                     (text_x - 5, text_y - text_size[1] - 5),
                                     (text_x + text_size[0] + 5, text_y + 5),
                                     (0, 0, 0), -1)
                        
                        cv2.putText(annotated_frame, missing_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    # Draw status above person
                    status_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    status_bg_y = y1 - 35 if (not result.get('is_compliant', True) and not is_observing) else y1 - 25
                    cv2.rectangle(annotated_frame,
                                 (x1 - 5, status_bg_y - status_size[1] - 5),
                                 (x1 + status_size[0] + 5, status_bg_y + 5),
                                 (0, 0, 0), -1)
                    cv2.putText(annotated_frame, status_text, (x1, status_bg_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            if enable_restricted_zone_module:
                # Draw restricted zone
                pts = np.array(restricted_zone, np.int32)
                cv2.polylines(annotated_frame, [pts], True, (0, 255, 255), 2)
                # Fill with semi-transparent overlay
                overlay = annotated_frame.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 255))
                cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
                cv2.putText(annotated_frame, "RESTRICTED ZONE", (restricted_zone[0][0], restricted_zone[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Highlight non-person objects in restricted zone
                for obj_id, obj_data in objects_in_zone.items():
                    x1, y1, x2, y2 = [int(coord) for coord in obj_data['bbox']]
                    time_in_zone = current_time - obj_data['first_seen']
                    
                    # Color based on time in zone
                    if time_in_zone >= alert_threshold:
                        color = (0, 0, 255)  # Red for alert
                        thickness = 3
                    else:
                        color = (0, 165, 255)  # Orange for warning
                        thickness = 2
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Create label with class name and time
                    class_name = obj_data.get('class_name', 'Unidentified Object')
                    label = f"{class_name} ({time_in_zone:.1f}s)"
                    
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add status text
            frame_count += 1
            status_text = f"Frame: {frame_count} | Press 'q' to quit"
            if enable_restricted_zone_module:
                status_text += " | 'r' to redraw zone"
            if enable_fire_module and fire_detected:
                status_text += f" | FIRE: {fire_count}"
                fire_detection_count += 1
            if enable_restricted_zone_module and len(objects_in_zone) > 0:
                status_text += f" | Objects in zone: {len(objects_in_zone)}"
            if ppe_model and len(compliance_results) > 0:
                num_non_compliant = sum(1 for r in compliance_results if r.get('is_compliant') is False)
                num_observing = sum(1 for r in compliance_results if r.get('is_observing', False))
                if num_non_compliant > 0:
                    status_text += f" | PPE Non-compliant: {num_non_compliant}"
                if num_observing > 0:
                    status_text += f" | Observing: {num_observing}"
            cv2.putText(
                annotated_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if ((enable_fire_module and fire_detected) or (enable_restricted_zone_module and len(unidentified_objects) > 0)) else (0, 255, 0),
                2
            )
            
            # Handle fire detection: save snapshot and send email
            if enable_fire_module and fire_detected and email_config:
                time_since_last_email = current_time - last_email_time
                
                if time_since_last_email >= email_cooldown:
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    snapshot_filename = detect_dir / f"fire_detection_{timestamp_str}_{save_count + 1}.jpg"
                    cv2.imwrite(str(snapshot_filename), annotated_frame)
                    save_count += 1
                    
                    print(f"\n🔥 FIRE DETECTED! Saving snapshot: {snapshot_filename.name}")
                    print(f"   Detections: {fire_count}, Max Confidence: {max_fire_confidence:.2%}")
                    
                    print("   Sending email alert...")
                    if send_fire_alert_email(email_config, str(snapshot_filename), fire_count, max_fire_confidence):
                        print("   Email sent successfully")
                        last_email_time = current_time
                    else:
                        print("   Failed to send email")

                    # ===== CSV LOGGING =====
                    append_event_to_csv(
                        camera=log_camera,
                        event_type="FIRE",
                        message=f"detections={fire_count}",
                        confidence=max_fire_confidence,
                        snapshot_path=str(snapshot_filename)
                    )
            
            # Handle unidentified objects in restricted zone
            if enable_restricted_zone_module and len(unidentified_objects) > 0 and email_config:
                time_since_last_email = current_time - last_unidentified_email_time
                
                if time_since_last_email >= email_cooldown:
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                    snapshot_filename = detect_dir / f"unidentified_object_{timestamp_str}_{save_count + 1}.jpg"
                    cv2.imwrite(str(snapshot_filename), annotated_frame)
                    save_count += 1
                    
                    print(f"\n⚠️ UNIDENTIFIED OBJECT(S) IN RESTRICTED ZONE!")
                    print(f"   Objects detected for >{alert_threshold} seconds:")
                    for obj in unidentified_objects:
                        print(f"   - {obj['class_name']} (Conf: {obj['confidence']:.2%})")
                    print(f"   Saving snapshot: {snapshot_filename.name}")
                    
                    print("   Sending email alert...")
                    if send_unidentified_object_email(email_config, str(snapshot_filename), unidentified_objects):
                        print("   Email sent successfully")
                        last_unidentified_email_time = current_time
                    else:
                        print("   Failed to send email")

                    # ===== CSV LOGGING =====
                    append_event_to_csv(
                        camera=log_camera,
                        event_type="RESTRICTED_ZONE_OBJECT",
                        message=f"objects={len(unidentified_objects)}",
                        snapshot_path=str(snapshot_filename)
                    )
            
            # Handle PPE non-compliance: save snapshot and send email
            # Only flag as non-compliant after observation period if mandatory PPE was never detected
            if ppe_model and len(compliance_results) > 0:
                non_compliant_persons = [r for r in compliance_results if r.get('is_compliant') is False]
                
                if len(non_compliant_persons) > 0 and email_config:
                    time_since_last_email = current_time - last_ppe_email_time
                    
                    if time_since_last_email >= email_cooldown:
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                        snapshot_filename = detect_dir / f"ppe_non_compliant_{timestamp_str}_{save_count + 1}.jpg"
                        cv2.imwrite(str(snapshot_filename), annotated_frame)
                        save_count += 1
                        
                        print(f"\n⚠️ PPE NON-COMPLIANCE DETECTED!")
                        print(f"   Non-compliant persons: {len(non_compliant_persons)}")
                        for i, person in enumerate(non_compliant_persons, 1):
                            missing = ', '.join(person['missing_ppe'])
                            print(f"   Person {i}: Missing - {missing}")
                        print(f"   Saving snapshot: {snapshot_filename.name}")
                        
                        print("   Sending email alert...")
                        if send_ppe_non_compliant_email(email_config, str(snapshot_filename), non_compliant_persons):
                            print("   Email sent successfully")
                            last_ppe_email_time = current_time
                        else:
                            print("   Failed to send email")

                        # ===== CSV LOGGING =====
                        append_event_to_csv(
                            camera=log_camera,
                            event_type="PPE_NON_COMPLIANT",
                            message=f"persons={len(non_compliant_persons)}",
                            snapshot_path=str(snapshot_filename)
                        )
            
            # Display the frame
            cv2.imshow(window_name, annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                if enable_restricted_zone_module:
                    # Redraw restricted zone
                    ret, setup_frame = cap.read()
                    if ret:
                        new_zone = draw_quadrilateral_interactive(setup_frame)
                        if new_zone is not None:
                            restricted_zone = new_zone
                            objects_in_zone = {}  # Reset tracking
                            persistent_objects = {}
                            person_ppe_tracking = {}  # Reset PPE person tracking
                            # Reinitialize background
                            background_initialized = False
                            background_frames = 0
                            restricted_zone_background = None
                            print("Restricted zone updated! Background will be recaptured from new zone...")
            elif key == ord('s'):
                # Save current frame manually
                save_count += 1
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = detect_dir / f"manual_snapshot_{timestamp_str}_{save_count}.jpg"
                cv2.imwrite(str(filename), annotated_frame)
                print(f"Frame saved as: {filename.name}")
            elif key == ord('c'):
                # Toggle confidence display
                show_conf = not show_conf
                print(f"Confidence display: {'ON' if show_conf else 'OFF'}")
            elif key == ord('l'):
                # Toggle labels display
                show_labels = not show_labels
                print(f"Labels display: {'ON' if show_labels else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        try:
            cv2.destroyWindow(window_name)
        except Exception:
            pass
        print("\n" + "="*50)
        print("Camera released. Detection stopped.")
        print(f"Total frames processed: {frame_count}")
        print(f"Total fire detections: {fire_detection_count}")
        if ppe_model:
            print(f"Total PPE non-compliant instances: {ppe_non_compliant_count}")
        print(f"Snapshots saved: {save_count}")
        print(f"Snapshot directory: {detect_dir.absolute()}")
        append_event_to_csv(
            camera=log_camera,
            event_type="SHUTDOWN",
            message=f"frames={frame_count}, fire={fire_detection_count}, ppe_non_compliant={ppe_non_compliant_count}, snapshots={save_count}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Live fire and smoke detection using webcam'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=os.path.join("models", "best.pt"),
        help='Path to the trained model (.pt file)'
    )
    parser.add_argument(
    '--camera',
    type=str,
    default="0",
    help='Camera index or RTSP URL'
)
    parser.add_argument(
        '--conf',
        type=float,
        default=0.40,
        help='Confidence threshold (0.0 to 1.0, default: 0.40)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide class labels'
    )
    parser.add_argument(
        '--no-conf',
        action='store_true',
        help='Hide confidence scores'
    )
    parser.add_argument(
        '--email-cooldown',
        type=int,
        default=60,
        help='Minimum seconds between email alerts (default: 60)'
    )
    parser.add_argument(
        '--use-camera-config',
        action='store_true',
        help='Run multi-camera mode using config/camera_config.json'
    )
    parser.add_argument(
        '--camera-config',
        type=str,
        default=os.path.join("config", "camera_config.json"),
        help='Path to camera configuration JSON file'
    )
    
    args = parser.parse_args()
    
    # Check for alternative model paths
    model_paths = [
        args.model,
        DEFAULT_FIRE_MODEL_PATH,
        os.path.join("models", "best.pt")
    ]
    
    model_path = None
    for path in model_paths:
        candidate_path = Path(path) if Path(path).is_absolute() else Path(BASE_DIR) / path
        if candidate_path.exists():
            model_path = candidate_path
            break
    
    if model_path is None:
        print("Error: Model file not found. Searched in:")
        for path in model_paths:
            candidate_path = Path(path) if Path(path).is_absolute() else Path(BASE_DIR) / path
            print(f"  - {candidate_path}")
        print("\nPlease specify the correct path using --model argument")
        exit(1)
    
    # ===== ADDED FOR MULTI-CAMERA SUPPORT =====
    shared_args = {
        "model_path": model_path,
        "conf_threshold": args.conf,
        "show_labels": not args.no_labels,
        "show_conf": not args.no_conf,
        "email_cooldown": args.email_cooldown
    }

    if args.use_camera_config:
        shared_models = load_global_models(model_path, enable_ppe=True)
        camera_config_path = Path(args.camera_config) if Path(args.camera_config).is_absolute() else Path(BASE_DIR) / args.camera_config
        run_multi_camera_from_config(str(camera_config_path), shared_args, shared_models)
    else:
        # Backward compatible single-camera mode
        detect_live(
            model_path=model_path,
            camera_index=args.camera,
            conf_threshold=args.conf,
            show_labels=not args.no_labels,
            show_conf=not args.no_conf,
            email_cooldown=args.email_cooldown
        )



