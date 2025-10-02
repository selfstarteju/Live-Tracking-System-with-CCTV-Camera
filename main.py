import cv2
import numpy as np
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    _HAS_PYZBAR = True
except Exception:
    pyzbar_decode = None
    _HAS_PYZBAR = False
from datetime import datetime
import database as db
import logging
import time
import os
import base64
from typing import Optional, Tuple, List
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("id_card_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize OCR reader (optional)
try:
    import easyocr
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        logger.info("OCR reader initialized")
    except Exception as e:
        reader = None
        logger.warning(f"easyocr installed but failed to initialize: {e}")
except Exception:
    reader = None
    logger.warning("easyocr not installed — OCR fallback disabled")

# Load face detection model (robust lookup)
face_cascade = None
cascade_name = 'haarcascade_frontalface_default.xml'
try:
    cascade_paths = []
    # Preferred location (newer OpenCV)
    if hasattr(cv2, 'data'):
        cascade_paths.append(cv2.data.haarcascades)
    # Try common package locations relative to cv2 module
    try:
        cv2_root = os.path.dirname(cv2.__file__)
        cascade_paths.append(os.path.join(cv2_root, 'data', 'haarcascades'))
        cascade_paths.append(os.path.join(cv2_root, 'haarcascades'))
    except Exception:
        pass
    # System locations
    cascade_paths.extend([
        '/usr/share/opencv/haarcascades',
        '/usr/local/share/opencv4/haarcascades',
        '/usr/local/share/opencv/haarcascades'
    ])

    found = False
    for p in cascade_paths:
        if not p:
            continue
        candidate = os.path.join(p, cascade_name)
        if os.path.exists(candidate):
            face_cascade = cv2.CascadeClassifier(candidate)
            logger.info(f"Loaded face cascade from: {candidate}")
            found = True
            break

    if not found:
        logger.warning("Face cascade not found in known locations; face detection disabled")
        face_cascade = None
except Exception as e:
    logger.warning(f"Error searching for face cascade: {e}; face detection disabled")
    face_cascade = None

class ImageProcessor:
    """Thread-safe image processor for handling captured images."""
    
    def __init__(self):
        self.processing_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_images, daemon=True)
        self.processing_thread.start()
    
    def _process_images(self):
        """Process images in a separate thread."""
        while True:
            try:
                task = self.processing_queue.get(timeout=1)
                if task is None:
                    break
                    
                task_type, data = task
                if task_type == 'save_image':
                    folder, filename, image = data
                    os.makedirs(folder, exist_ok=True)
                    cv2.imwrite(os.path.join(folder, filename), image)
                elif task_type == 'save_to_db':
                    barcode_data, timestamp, full_image, card_image, face_image = data
                    db.save_scan_images(barcode_data, timestamp, full_image, card_image, face_image)
                    
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
    
    def save_image(self, folder, filename, image):
        """Add an image to the processing queue."""
        self.processing_queue.put(('save_image', (folder, filename, image)))
    
    def save_to_db(self, barcode_data, timestamp, full_image, card_image, face_image):
        """Add images to be saved to the database."""
        self.processing_queue.put(('save_to_db', (barcode_data, timestamp, full_image, card_image, face_image)))

class IDCardTracker:
    def __init__(self, save_frames: bool = False, output_folder: str = 'output'):
        """
        Initialize the ID Card Tracker.
        
        Args:
            save_frames: Whether to save processed frames
            output_folder: Folder to save output frames
        """
        self.save_frames = save_frames
        self.output_folder = output_folder
        self.image_processor = ImageProcessor()
        
        # Create output folder if it doesn't exist
        if self.save_frames and not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Initialize webcam or video source (robust)
        self.cap = None
        cam_source = os.environ.get('CAMERA_SOURCE')  # can be index like '0' or path to file

        def try_open(source, backend=None):
            try:
                if backend is not None:
                    cap = cv2.VideoCapture(source, backend)
                else:
                    cap = cv2.VideoCapture(source)
                if cap is not None and cap.isOpened():
                    return cap
            except Exception:
                pass
            return None

        # If CAMERA_SOURCE provided, try it first
        if cam_source:
            # if it's numeric, try as index
            if cam_source.isdigit():
                idx = int(cam_source)
                self.cap = try_open(idx, cv2.CAP_V4L2) or try_open(idx)
            else:
                # try as file path
                if os.path.exists(cam_source):
                    self.cap = try_open(cam_source)
                else:
                    logger.warning(f"CAMERA_SOURCE provided but not found: {cam_source}")

        # Try common camera indices/backends if not opened yet
        if self.cap is None or not self.cap.isOpened():
            for idx in range(0, 5):
                self.cap = try_open(idx, cv2.CAP_V4L2) or try_open(idx)
                if self.cap is not None and self.cap.isOpened():
                    logger.info(f"Opened camera at index {idx}")
                    break

        if self.cap is None or not self.cap.isOpened():
            logger.error("Could not open any camera. If you're on Linux, try setting CAMERA_SOURCE to /dev/video0 or install v4l2/ffmpeg backends.")
            raise ValueError("Could not open webcam or video source")

        # Optionally set desired resolution
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except Exception:
            pass

        # Get actual webcam resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        logger.info(f"Webcam initialized with resolution: {self.width}x{self.height}")
        
        # Initialize frame counter
        self.frame_count = 0
        
        # Last detected barcodes to avoid duplicate scanning
        self.detected_barcodes = {}  # Dictionary to track recently detected barcodes
        self.detection_cooldown = 5  # seconds
        
        # Create folders for storing captured images
        self.scans_folder = "scans"
        os.makedirs(self.scans_folder, exist_ok=True)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Preprocessed frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edged = cv2.Canny(blur, 30, 200)
        
        # Dilate edges to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        
        return edged

    # --- Helpers for perspective transform and barcode enhancement ---
    def order_points(self, pts: np.ndarray) -> np.ndarray:
        # pts: (4,2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        # compute the height of the new image
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def pad_box(self, pts: np.ndarray, pad_ratio: float = 0.12) -> np.ndarray:
        """Expand a 4-point box outward from its center by pad_ratio (fraction of width/height).

        Args:
            pts: array shape (4,2)
            pad_ratio: fractional padding (0.1 = 10%)

        Returns:
            padded pts as (4,2) float32
        """
        rect = self.order_points(pts.astype('float32'))
        (tl, tr, br, bl) = rect
        # center
        cx = np.mean(rect[:, 0])
        cy = np.mean(rect[:, 1])
        # vector from center to each corner
        vecs = rect - np.array([cx, cy])
        # scale vectors
        scaled = rect + vecs * pad_ratio
        return scaled.astype('float32')

    def enhance_card_image(self, card_img: np.ndarray) -> np.ndarray:
        # Convert to grayscale and try to improve contrast
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # Resize to a minimum width for better barcode detection
        h, w = gray.shape[:2]
        # Increase minimum target width to help barcode decoders by providing more pixels
        target_w = max(800, w)
        if w < target_w:
            scale = target_w / float(w)
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        return gray

    def try_read_barcode_variants(self, img: np.ndarray) -> List[str]:
        # Try multiple preprocessing strategies to decode barcodes
        results = []

        def try_decode(image):
            try:
                if _HAS_PYZBAR and pyzbar_decode is not None:
                    # pyzbar sometimes performs better on 3-channel images; try both
                    decs = pyzbar_decode(image)
                    if not decs:
                        try:
                            if len(image.shape) == 2:
                                img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                                decs = pyzbar_decode(img_color)
                        except Exception:
                            pass
                    return [d.data.decode('utf-8') for d in decs]
                else:
                    # Fallback: use easyocr to extract likely barcode-like text (alphanumeric)
                    try:
                        txts = reader.readtext(image)
                        candidates = []
                        for res in txts:
                            text = res[1].strip()
                            # Filter short or low-confidence results
                            if len(text) >= 4 and res[2] > 0.4:
                                # keep alphanumeric-like strings
                                if any(c.isalnum() for c in text):
                                    candidates.append(text)
                        return candidates
                    except Exception:
                        return []
            except Exception:
                return []

        # 1) original
        res = try_decode(img)
        if res:
            logger.debug(f"try_read_barcode_variants: original success: {res}")
        results.extend(res)

        # 2) CLAHE (adaptive histogram equalization)
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img_clahe = clahe.apply(img)
            res = try_decode(img_clahe)
            if res:
                logger.debug(f"try_read_barcode_variants: CLAHE success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 3) Bilateral filter to preserve edges
        try:
            bilat = cv2.bilateralFilter(img, 9, 75, 75)
            res = try_decode(bilat)
            if res:
                logger.debug(f"try_read_barcode_variants: bilateral success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 4) median blur + morphological close
        try:
            med = cv2.medianBlur(img, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            closed = cv2.morphologyEx(med, cv2.MORPH_CLOSE, kernel)
            res = try_decode(closed)
            if res:
                logger.debug(f"try_read_barcode_variants: closed success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 5) resized (larger scales)
        try:
            for scale in [1.5, 2.0, 3.0, 4.0]:
                img_big = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
                res = try_decode(img_big)
                if res:
                    logger.debug(f"try_read_barcode_variants: resized x{scale} success: {res}")
                results.extend(res)
        except Exception:
            pass

        # 6) adaptive threshold
        try:
            th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
            res = try_decode(th)
            if res:
                logger.debug(f"try_read_barcode_variants: adaptive thresh success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 7) Otsu threshold
        try:
            _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            res = try_decode(otsu)
            if res:
                logger.debug(f"try_read_barcode_variants: otsu success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 8) rotations
        try:
            for angle in [90, 180, 270]:
                M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
                rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                res = try_decode(rot)
                if res:
                    logger.debug(f"try_read_barcode_variants: rot {angle} success: {res}")
                results.extend(res)
        except Exception:
            pass

        # 9) invert colors (sometimes barcodes are light-on-dark)
        try:
            inv = cv2.bitwise_not(img)
            res = try_decode(inv)
            if res:
                logger.debug(f"try_read_barcode_variants: invert success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 10) vertical gradient (Sobel) to emphasize barcode stripes
        try:
            sobel = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
            sobel = cv2.convertScaleAbs(sobel)
            # normalize and threshold a bit
            sobel_norm = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
            _, sobel_th = cv2.threshold(sobel_norm, 30, 255, cv2.THRESH_BINARY)
            res = try_decode(sobel_th)
            if res:
                logger.debug(f"try_read_barcode_variants: sobel success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 11) stronger morphological close after otsu to join barcode bars
        try:
            _, otsu2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,3))
            closed2 = cv2.morphologyEx(otsu2, cv2.MORPH_CLOSE, kernel)
            res = try_decode(closed2)
            if res:
                logger.debug(f"try_read_barcode_variants: strong close success: {res}")
            results.extend(res)
        except Exception:
            pass

        # 12) crop central vertical band (barcodes often sit in middle of card) and try upscale/thresholds
        try:
            h, w = img.shape[:2]
            cx1 = int(w * 0.2)
            cx2 = int(w * 0.8)
            central = img[:, cx1:cx2]
            for scale in [2.0, 3.0, 4.0]:
                big = cv2.resize(central, (int(central.shape[1]*scale), int(central.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
                # try CLAHE + adaptive thresh combo
                try:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    bigc = clahe.apply(big)
                except Exception:
                    bigc = big
                try:
                    th = cv2.adaptiveThreshold(bigc, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8)
                except Exception:
                    _, th = cv2.threshold(bigc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                res = try_decode(th)
                if res:
                    logger.debug(f"try_read_barcode_variants: central x{scale} success: {res}")
                results.extend(res)
        except Exception:
            pass

        # Deduplicate
        unique = list(dict.fromkeys(results))
        # 13) As a last resort, try localized region proposals based on vertical gradients
        if not unique:
            try:
                regions = self.locate_barcode_regions(img)
                for (x, y, w, h), crop in regions:
                    # try a few upscales on crop
                    for scale in [1.5, 2.5, 4.0]:
                        try:
                            big = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)), interpolation=cv2.INTER_LINEAR)
                        except Exception:
                            big = crop
                        r = try_decode(big)
                        if r:
                            logger.debug(f"try_read_barcode_variants: region ({x},{y},{w},{h}) x{scale} success: {r}")
                        results.extend(r)
                unique = list(dict.fromkeys(results))
            except Exception:
                pass

        return unique

    def locate_barcode_regions(self, gray: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], np.ndarray]]:
        """Locate likely barcode regions by emphasizing vertical strokes and finding narrow, wide contours.

        Returns list of ((x,y,w,h), crop) tuples.
        """
        regions = []
        try:
            if len(gray.shape) == 3:
                g = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            else:
                g = gray

            # compute vertical gradient (Sobel x)
            gradx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
            gradx = cv2.convertScaleAbs(gradx)
            # normalize and threshold
            _, th = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # close to join bars
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,3))
            closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

            # find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h, w = g.shape[:2]
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if area < 500 or cw < 40:
                    continue
                aspect = cw / float(max(1, ch))
                # barcodes usually are wider than tall, choose aspect > 2.0
                if aspect < 2.0:
                    continue

                # expand slightly
                padx = int(cw * 0.12)
                pady = int(ch * 0.12)
                x0 = max(0, x - padx)
                y0 = max(0, y - pady)
                x1 = min(w, x + cw + padx)
                y1 = min(h, y + ch + pady)
                crop = g[y0:y1, x0:x1]
                regions.append(((x0, y0, x1-x0, y1-y0), crop))

            # sort by area descending
            regions = sorted(regions, key=lambda r: r[0][2]*r[0][3], reverse=True)
        except Exception as e:
            logger.debug(f"locate_barcode_regions error: {e}")

        return regions
    
    def detect_id_card(self, frame: np.ndarray, preprocessed: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Detect ID card in the frame.
        
        Args:
            frame: Original frame
            preprocessed: Preprocessed frame (optional)
            
        Returns:
            Contour of the detected card or None
        """
        if preprocessed is None:
            preprocessed = self.preprocess_frame(frame)
            
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_contour = None
        max_area = 0
        
        # Use minAreaRect to detect rotated rectangles as well
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3000:
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(int)

            # compute aspect ratio and area
            w = np.linalg.norm(box[0] - box[1])
            h = np.linalg.norm(box[1] - box[2])
            if h == 0 or w == 0:
                continue
            aspect_ratio = max(w, h) / float(min(w, h))

            # standard card aspect ~1.3-1.9 depending on orientation; area threshold
            if 1.2 < aspect_ratio < 2.5 and area > max_area:
                max_area = area
                card_contour = box.reshape((4,1,2))
        
        return card_contour
    
    def read_barcodes(self, image: np.ndarray) -> List[str]:
        """
        Read all barcodes from image.
        
        Args:
            image: Input image
            
        Returns:
            List of decoded barcode data
        """
        try:
            # Ensure grayscale input for pyzbar when appropriate
            img = image
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            results = self.try_read_barcode_variants(img)
            return results
        except Exception as e:
            logger.error(f"Error reading barcodes: {str(e)}")
            return []
    
    def read_text(self, image: np.ndarray) -> str:
        """
        Read text from image using OCR.
        
        Args:
            image: Input image
            
        Returns:
            Extracted text
        """
        try:
            if reader is None:
                return ""
            text_results = reader.readtext(image)
            extracted_text = " ".join([res[1] for res in text_results if res[2] > 0.5])  # Only use high confidence results
            return extracted_text
        except Exception as e:
            logger.error(f"Error reading text: {str(e)}")
            return ""
    
    def detect_color_indicator(self, image: np.ndarray) -> Optional[str]:
        """
        Detect color indicator in the corner of the card.
        
        Args:
            image: Input image
            
        Returns:
            Detected color name or None
        """
        try:
            height, width = image.shape[:2]
            
            # Check top-right corner
            roi = image[0:height//4, 3*width//4:width]
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Define color ranges in HSV
            colors = {
                "green": ((36, 25, 25), (86, 255, 255)),
                "yellow": ((20, 100, 100), (40, 255, 255)),
                "red": ((0, 100, 100), (10, 255, 255)),
                "blue": ((100, 100, 100), (130, 255, 255))
            }
            
            detected_color = None
            max_pixels = 0
            
            for color_name, (lower, upper) in colors.items():
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                pixel_count = cv2.countNonZero(mask)
                
                if pixel_count > 100 and pixel_count > max_pixels:
                    max_pixels = pixel_count
                    detected_color = color_name
            
            return detected_color
        except Exception as e:
            logger.error(f"Error detecting color: {str(e)}")
            return None
    
    def detect_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect faces in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of face ROIs
        """
        try:
            if face_cascade is None:
                return []

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            face_images = []
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                face_images.append(face_roi)

            return face_images
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []
    
    def process_barcode_detection(self, frame: np.ndarray, barcode: str, card_contour=None, card_roi=None):
        """
        Process a detected barcode by capturing and saving images.
        
        Args:
            frame: Original frame
            barcode: Detected barcode data
            card_contour: Contour of the ID card (if detected)
            card_roi: ROI of the ID card (if detected)
        """
        current_time = time.time()
        
        # Check if this barcode was recently detected
        if barcode in self.detected_barcodes:
            last_detection_time = self.detected_barcodes[barcode]
            if (current_time - last_detection_time) < self.detection_cooldown:
                return False  # Skip this detection to avoid duplicates
        
        # Update the last detection time for this barcode
        self.detected_barcodes[barcode] = current_time
        
        # Clean up old barcode records
        old_barcodes = [bc for bc, dt in self.detected_barcodes.items() 
                        if (current_time - dt) > self.detection_cooldown * 2]
        for bc in old_barcodes:
            del self.detected_barcodes[bc]
        
        # Generate timestamp for filenames
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Create folder for this scan
        scan_folder = os.path.join(self.scans_folder, f"{barcode}_{timestamp_str}")
        os.makedirs(scan_folder, exist_ok=True)
        
        # Save full frame
        full_frame_path = os.path.join(scan_folder, "full_frame.jpg")
        cv2.imwrite(full_frame_path, frame)
        
        # Save card image if available
        card_image = None
        if card_roi is not None:
            card_image_path = os.path.join(scan_folder, "id_card.jpg")
            cv2.imwrite(card_image_path, card_roi)
            card_image = card_roi
        
        # Detect faces in the frame (to verify identity)
        face_images = self.detect_faces(frame)
        for i, face_image in enumerate(face_images):
            face_path = os.path.join(scan_folder, f"face_{i+1}.jpg")
            cv2.imwrite(face_path, face_image)
        
        # Read text and detect color if card is available
        text = ""
        color = None
        if card_roi is not None:
            text = self.read_text(card_roi)
            color = self.detect_color_indicator(card_roi)

        # If we detected at least one face, mark attendance as verified
        verified = 1 if len(face_images) > 0 else 0
        
        # Log to database with image paths
        db.log_attendance_with_images(
            barcode=barcode,
            name=text,
            color=color,
            timestamp=timestamp,
            full_frame_path=full_frame_path,
            card_image_path=card_image_path if card_image else None,
            face_image_paths=[os.path.join(scan_folder, f"face_{i+1}.jpg") for i in range(len(face_images))],
            verified=verified
        )
        
        logger.info(f"Processed barcode: {barcode}, Text: {text}, Color: {color}")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame
        """
        processed_frame = frame.copy()
        
        # First try to read barcodes directly from the frame
        barcodes = self.read_barcodes(frame)
        
        processed_barcodes = []  # Track which barcodes were processed in this frame
        
        if barcodes:
            # If pyzbar is available we can get polygon points; otherwise use read_barcodes results
            if _HAS_PYZBAR and pyzbar_decode is not None:
                try:
                    decoded = pyzbar_decode(frame)
                except Exception:
                    decoded = []

                for barcode_data in decoded:
                    barcode = barcode_data.data.decode('utf-8')

                    # Draw barcode bounding box if polygon exists
                    try:
                        points = barcode_data.polygon
                        if len(points) > 4:
                            hull = cv2.convexHull(np.array([point for point in points], dtype=np.int32))
                            cv2.drawContours(processed_frame, [hull], 0, (0, 255, 0), 2)
                        else:
                            cv2.drawContours(processed_frame, [np.array(points, dtype=np.int32)], 0, (0, 255, 0), 2)
                    except Exception:
                        pass

                    if self.process_barcode_detection(processed_frame, barcode):
                        processed_barcodes.append(barcode)
                        cv2.putText(processed_frame, f"Barcode: {barcode} - Processed",
                                    (10, 30 + len(processed_barcodes) * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(processed_frame, f"Barcode: {barcode} - Duplicate",
                                    (10, 30 + len(processed_barcodes) * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Fallback: we have list of barcode strings from read_barcodes
                for i, barcode in enumerate(barcodes):
                    if self.process_barcode_detection(processed_frame, barcode):
                        processed_barcodes.append(barcode)
                        cv2.putText(processed_frame, f"Barcode: {barcode} - Processed",
                                    (10, 30 + len(processed_barcodes) * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(processed_frame, f"Barcode: {barcode} - Duplicate",
                                    (10, 30 + len(processed_barcodes) * 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # If no barcodes were processed, try to detect ID cards
        if not processed_barcodes:
            # Try to detect ID card
            preprocessed = self.preprocess_frame(frame)
            card_contour = self.detect_id_card(frame, preprocessed)
            
            if card_contour is not None:
                # Get bounding rectangle
                # Use perspective transform to get a straight card view
                try:
                    pts = card_contour.reshape(4, 2).astype('float32')
                    # pad the detected box slightly before warping to include surrounding barcode area
                    try:
                        pts_padded = self.pad_box(pts, pad_ratio=0.18)
                    except Exception:
                        pts_padded = pts
                    warped = self.four_point_transform(frame, pts_padded)
                except Exception:
                    # Fallback to bounding rect
                    x, y, w, h = cv2.boundingRect(card_contour)
                    warped = frame[y:y+h, x:x+w]

                # Enhance the warped card image for barcode reading
                enhanced = self.enhance_card_image(warped)

                # Try to read barcodes using multiple strategies
                card_barcodes = self.try_read_barcode_variants(enhanced)

                if card_barcodes:
                    # Draw card contour
                    cv2.drawContours(processed_frame, [card_contour], -1, (0, 255, 0), 2)

                    # Process each detected barcode
                    offset_x, offset_y = 0, 0
                    try:
                        x, y, w, h = cv2.boundingRect(card_contour)
                        offset_x, offset_y = x, y
                    except Exception:
                        pass

                    for i, barcode in enumerate(card_barcodes):
                        if self.process_barcode_detection(processed_frame, barcode, card_contour, warped):
                            processed_barcodes.append(barcode)
                            # Add text to frame
                            cv2.putText(processed_frame, f"Card Barcode: {barcode} - Processed",
                                        (offset_x, offset_y - 10 - i * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            cv2.putText(processed_frame, f"Card Barcode: {barcode} - Duplicate",
                                        (offset_x, offset_y - 10 - i * 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Draw card contour but no barcode found
                    try:
                        x, y, w, h = cv2.boundingRect(card_contour)
                        cv2.drawContours(processed_frame, [card_contour], -1, (0, 255, 255), 2)
                        cv2.putText(processed_frame, "Card detected, no barcode", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    except Exception:
                        cv2.putText(processed_frame, "Card detected, no barcode", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # Optionally save warped/enhanced image for debugging
                    try:
                        if os.environ.get('DEBUG_SAVE_FAILED') == '1':
                            dbg_dir = os.path.join('debug_failed')
                            os.makedirs(dbg_dir, exist_ok=True)
                            ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                            if 'warped' in locals():
                                cv2.imwrite(os.path.join(dbg_dir, f'warped_{ts}.jpg'), warped)
                                cv2.imwrite(os.path.join(dbg_dir, f'enhanced_{ts}.jpg'), enhanced)
                    except Exception as e:
                        logger.debug(f"Failed to save debug images: {e}")
        
        # Display count of processed barcodes in this frame
        if processed_barcodes:
            cv2.putText(processed_frame, f"Processed {len(processed_barcodes)} barcode(s) in this frame", 
                        (10, self.height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return processed_frame
    
    def run(self):
        """
        Run the ID card tracking process.
        """
        logger.info("Starting ID card tracking...")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame from webcam")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display frame
                cv2.imshow("ID Card Tracker", processed_frame)
                
                # Save frame if enabled
                if self.save_frames:
                    self.image_processor.save_image(
                        self.output_folder, 
                        f"frame_{self.frame_count:06d}.jpg", 
                        processed_frame
                    )
                    self.frame_count += 1
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested to stop")
                    break
                
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("ID card tracking stopped")

if __name__ == "__main__":
    # Initialize database
    db.init_db()
    
    # Create and run tracker
    tracker = IDCardTracker(save_frames=True)
    tracker.run()