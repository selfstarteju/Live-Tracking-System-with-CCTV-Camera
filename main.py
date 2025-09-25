import cv2
import numpy as np
from pyzbar.pyzbar import decode
import easyocr
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

# Initialize OCR reader
try:
    reader = easyocr.Reader(['en'], gpu=False)
    logger.info("OCR reader initialized")
except Exception as e:
    logger.error(f"Failed to initialize OCR reader: {str(e)}")
    exit(1)

# Load face detection model
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    logger.info("Face detection model loaded")
except Exception as e:
    logger.error(f"Failed to load face detection model: {str(e)}")
    exit(1)

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
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)  # 0 is the default webcam
        if not self.cap.isOpened():
            logger.error("Could not open webcam")
            raise ValueError("Could not open webcam")
        
        # Set webcam resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Get actual webcam resolution
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:  # Filter small contours
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                
                # Check if contour has 4 points (rectangle) and is large enough
                if len(approx) == 4 and area > max_area:
                    # Additional check for aspect ratio (ID cards typically have specific ratios)
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    
                    # Standard ID card aspect ratio is approximately 1.586 (85.6mm x 54mm)
                    if 1.3 < aspect_ratio < 1.8:
                        max_area = area
                        card_contour = approx
        
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
            barcodes = decode(image)
            return [barcode.data.decode('utf-8') for barcode in barcodes]
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
        
        # Detect and save faces
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
        
        # Log to database with image paths
        db.log_attendance_with_images(
            barcode=barcode,
            name=text,
            color=color,
            timestamp=timestamp,
            full_frame_path=full_frame_path,
            card_image_path=card_image_path if card_image else None,
            face_image_paths=[os.path.join(scan_folder, f"face_{i+1}.jpg") for i in range(len(face_images))]
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
            # Draw all detected barcodes
            for barcode_data in decode(frame):
                barcode = barcode_data.data.decode('utf-8')
                
                # Draw barcode bounding box
                points = barcode_data.polygon
                
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.int32))
                    cv2.drawContours(processed_frame, [hull], 0, (0, 255, 0), 2)
                else:
                    cv2.drawContours(processed_frame, [np.array(points, dtype=np.int32)], 0, (0, 255, 0), 2)
                
                # Process the barcode detection
                if self.process_barcode_detection(processed_frame, barcode):
                    processed_barcodes.append(barcode)
                    # Add text to frame
                    cv2.putText(processed_frame, f"Barcode: {barcode} - Processed", 
                                (10, 30 + len(processed_barcodes) * 30),  # Offset text for multiple barcodes
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Add text to frame
                    cv2.putText(processed_frame, f"Barcode: {barcode} - Duplicate", 
                                (10, 30 + len(processed_barcodes) * 30),  # Offset text for multiple barcodes
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # If no barcodes were processed, try to detect ID cards
        if not processed_barcodes:
            # Try to detect ID card
            preprocessed = self.preprocess_frame(frame)
            card_contour = self.detect_id_card(frame, preprocessed)
            
            if card_contour is not None:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(card_contour)
                
                # Extract card ROI
                card_roi = frame[y:y+h, x:x+w]
                
                # Try to read barcodes from card ROI
                card_barcodes = self.read_barcodes(card_roi)
                
                if card_barcodes:
                    # Draw card contour
                    cv2.drawContours(processed_frame, [card_contour], -1, (0, 255, 0), 2)
                    
                    # Process each detected barcode
                    for i, barcode in enumerate(card_barcodes):
                        if self.process_barcode_detection(processed_frame, barcode, card_contour, card_roi):
                            processed_barcodes.append(barcode)
                            # Add text to frame
                            cv2.putText(processed_frame, f"Card Barcode: {barcode} - Processed", 
                                        (x, y - 10 - i * 30),  # Offset text for multiple barcodes
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        else:
                            # Add text to frame
                            cv2.putText(processed_frame, f"Card Barcode: {barcode} - Duplicate", 
                                        (x, y - 10 - i * 30),  # Offset text for multiple barcodes
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # Draw card contour but no barcode found
                    cv2.drawContours(processed_frame, [card_contour], -1, (0, 255, 255), 2)
                    cv2.putText(processed_frame, "Card detected, no barcode", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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