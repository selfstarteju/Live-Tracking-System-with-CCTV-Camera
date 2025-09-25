import sqlite3
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import os
from contextlib import contextmanager
import base64

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """A class to manage database operations for the ID Card Tracking System."""
    
    def __init__(self, db_path: str = 'attendance.db'):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_exists()
        
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {str(e)}")
            raise
        finally:
            conn.close()
    
    def _ensure_db_exists(self):
        """Ensure the database and tables exist."""
        if not os.path.exists(self.db_path):
            logger.info(f"Database not found. Creating new database at {self.db_path}")
            self.init_db()
        else:
            # Check if tables exist
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = {row[0] for row in cursor.fetchall()}
                
                if 'attendance' not in tables:
                    logger.info("Attendance table not found. Creating it.")
                    self._create_attendance_table(conn)
                
                if 'scan_images' not in tables:
                    logger.info("Scan images table not found. Creating it.")
                    self._create_scan_images_table(conn)
    
    def init_db(self):
        """Initialize the database with required tables."""
        with self._get_connection() as conn:
            self._create_attendance_table(conn)
            self._create_scan_images_table(conn)
            conn.commit()
            logger.info("Database initialized successfully")
    
    def _create_attendance_table(self, conn):
        """Create the attendance table."""
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                barcode TEXT NOT NULL,
                name TEXT NOT NULL,
                color TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        ''')
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_barcode ON attendance(barcode)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON attendance(timestamp)')
    
    def _create_scan_images_table(self, conn):
        """Create the scan images table."""
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                barcode TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                full_frame_path TEXT NOT NULL,
                card_image_path TEXT,
                face_image_paths TEXT,  -- JSON array of paths
                created_at TEXT NOT NULL
            )
        ''')
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_barcode ON scan_images(barcode)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_timestamp ON scan_images(timestamp)')
    
    def log_attendance(self, barcode: str, name: str, color: str, timestamp: datetime) -> bool:
        """
        Log attendance data to the database.
        
        Args:
            barcode: The barcode data from the ID card
            name: The name extracted from the ID card
            color: The color indicator status
            timestamp: The timestamp of the detection
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO attendance (barcode, name, color, timestamp) VALUES (?, ?, ?, ?)",
                    (barcode, name, color, timestamp.isoformat())
                )
                conn.commit()
                logger.info(f"Attendance logged: {barcode} - {color}")
                return True
        except Exception as e:
            logger.error(f"Failed to log attendance: {str(e)}")
            return False
    
    def log_attendance_with_images(self, barcode: str, name: str, color: str, 
                                  timestamp: datetime, full_frame_path: str,
                                  card_image_path: Optional[str] = None,
                                  face_image_paths: Optional[List[str]] = None) -> bool:
        """
        Log attendance data with image paths to the database.
        
        Args:
            barcode: The barcode data from the ID card
            name: The name extracted from the ID card
            color: The color indicator status
            timestamp: The timestamp of the detection
            full_frame_path: Path to the full frame image
            card_image_path: Path to the cropped ID card image
            face_image_paths: List of paths to detected face images
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Log attendance data
            self.log_attendance(barcode, name, color, timestamp)
            
            # Log image paths
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO scan_images (barcode, timestamp, full_frame_path, card_image_path, face_image_paths, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        barcode,
                        timestamp.isoformat(),
                        full_frame_path,
                        card_image_path,
                        json.dumps(face_image_paths) if face_image_paths else None,
                        datetime.now().isoformat()
                    )
                )
                conn.commit()
                logger.info(f"Attendance with images logged: {barcode}")
                return True
        except Exception as e:
            logger.error(f"Failed to log attendance with images: {str(e)}")
            return False
    
    def save_scan_images(self, barcode: str, timestamp: datetime, 
                        full_image: bytes, card_image: Optional[bytes] = None,
                        face_images: Optional[List[bytes]] = None) -> bool:
        """
        Save scan images to the database as BLOBs.
        
        Args:
            barcode: The barcode data
            timestamp: The timestamp of the scan
            full_image: Full frame image as bytes
            card_image: Cropped ID card image as bytes
            face_images: List of face images as bytes
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create folder for this scan
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            scan_folder = os.path.join("scans", f"{barcode}_{timestamp_str}")
            os.makedirs(scan_folder, exist_ok=True)
            
            # Save images to files
            full_frame_path = os.path.join(scan_folder, "full_frame.jpg")
            with open(full_frame_path, 'wb') as f:
                f.write(full_image)
            
            card_image_path = None
            if card_image:
                card_image_path = os.path.join(scan_folder, "id_card.jpg")
                with open(card_image_path, 'wb') as f:
                    f.write(card_image)
            
            face_image_paths = []
            if face_images:
                for i, face_image in enumerate(face_images):
                    face_path = os.path.join(scan_folder, f"face_{i+1}.jpg")
                    with open(face_path, 'wb') as f:
                        f.write(face_image)
                    face_image_paths.append(face_path)
            
            # Log to database with image paths
            return self.log_attendance_with_images(
                barcode=barcode,
                name="",  # Will be filled in by the main process
                color="",  # Will be filled in by the main process
                timestamp=timestamp,
                full_frame_path=full_frame_path,
                card_image_path=card_image_path,
                face_image_paths=face_image_paths
            )
        except Exception as e:
            logger.error(f"Failed to save scan images: {str(e)}")
            return False
    
    def get_scan_images(self, barcode: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get scan images for a specific barcode.
        
        Args:
            barcode: The barcode to search for
            limit: Maximum number of records to return
            
        Returns:
            List of scan image records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM scan_images WHERE barcode = ? ORDER BY timestamp DESC LIMIT ?",
                    (barcode, limit)
                )
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    result = dict(row)
                    # Parse face image paths JSON
                    if result['face_image_paths']:
                        import json
                        result['face_image_paths'] = json.loads(result['face_image_paths'])
                    else:
                        result['face_image_paths'] = []
                    results.append(result)
                
                return results
        except Exception as e:
            logger.error(f"Failed to get scan images: {str(e)}")
            return []
    
    def get_image_as_base64(self, image_path: str) -> Optional[str]:
        """
        Get an image as a base64-encoded string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded image or None if file not found
        """
        try:
            if not os.path.exists(image_path):
                return None
                
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded_string}"
        except Exception as e:
            logger.error(f"Failed to encode image: {str(e)}")
            return None
    
    def fetch_attendance(self, date_filter: Optional[str] = None, 
                        color_filter: Optional[str] = None,
                        limit: Optional[int] = None,
                        offset: Optional[int] = 0) -> List[Dict[str, Any]]:
        """
        Fetch attendance records from the database.
        
        Args:
            date_filter: Filter by date (YYYY-MM-DD format)
            color_filter: Filter by color status
            limit: Maximum number of records to return
            offset: Number of records to skip
            
        Returns:
            List of attendance records as dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query
                query = "SELECT * FROM attendance"
                params = []
                
                # Add filters
                conditions = []
                if date_filter:
                    conditions.append("date(timestamp) = ?")
                    params.append(date_filter)
                
                if color_filter and color_filter != 'all':
                    conditions.append("color = ?")
                    params.append(color_filter)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                # Add ordering and pagination
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                if offset:
                    query += " OFFSET ?"
                    params.append(offset)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to dictionaries
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch attendance: {str(e)}")
            return []
    
    def get_attendance_stats(self, days: int = 7) -> Dict[str, Any]:
        """
        Get attendance statistics for the specified number of days.
        
        Args:
            days: Number of days to include in the statistics
            
        Returns:
            Dictionary containing attendance statistics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Calculate date range
                end_date = datetime.now()
                start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Get total records
                cursor.execute(
                    "SELECT COUNT(*) FROM attendance WHERE timestamp >= ? AND timestamp <= ?",
                    (start_date.isoformat(), end_date.isoformat())
                )
                total_records = cursor.fetchone()[0]
                
                # Get counts by color
                color_counts = {}
                for color in ['green', 'yellow', 'red', 'blue']:
                    cursor.execute(
                        "SELECT COUNT(*) FROM attendance WHERE color = ? AND timestamp >= ? AND timestamp <= ?",
                        (color, start_date.isoformat(), end_date.isoformat())
                    )
                    color_counts[color] = cursor.fetchone()[0]
                
                # Get counts by day
                daily_counts = {}
                for i in range(days):
                    day = start_date.replace(day=start_date.day - i)
                    next_day = day.replace(day=day.day + 1)
                    
                    cursor.execute(
                        "SELECT COUNT(*) FROM attendance WHERE timestamp >= ? AND timestamp < ?",
                        (day.isoformat(), next_day.isoformat())
                    )
                    daily_counts[day.strftime('%Y-%m-%d')] = cursor.fetchone()[0]
                
                return {
                    'total_records': total_records,
                    'by_color': color_counts,
                    'by_day': daily_counts
                }
        except Exception as e:
            logger.error(f"Failed to get attendance stats: {str(e)}")
            return {
                'total_records': 0,
                'by_color': {},
                'by_day': {}
            }
    
    def get_attendance_by_barcode(self, barcode: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get attendance records for a specific barcode.
        
        Args:
            barcode: The barcode to search for
            limit: Maximum number of records to return
            
        Returns:
            List of attendance records as dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM attendance WHERE barcode = ? ORDER BY timestamp DESC LIMIT ?",
                    (barcode, limit)
                )
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get attendance by barcode: {str(e)}")
            return []
    
    def delete_attendance_record(self, record_id: int) -> bool:
        """
        Delete an attendance record by ID.
        
        Args:
            record_id: The ID of the record to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM attendance WHERE id = ?", (record_id,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete attendance record: {str(e)}")
            return False

# Import json for face_image_paths
import json

# Create a global instance for backward compatibility
db_manager = DatabaseManager()

# Backward compatibility functions
def init_db():
    """Initialize the database (backward compatibility)."""
    db_manager.init_db()

def log_attendance(barcode: str, name: str, color: str, timestamp: datetime) -> bool:
    """Log attendance data (backward compatibility)."""
    return db_manager.log_attendance(barcode, name, color, timestamp)

def log_attendance_with_images(barcode: str, name: str, color: str, 
                              timestamp: datetime, full_frame_path: str,
                              card_image_path: Optional[str] = None,
                              face_image_paths: Optional[List[str]] = None) -> bool:
    """Log attendance data with image paths (backward compatibility)."""
    return db_manager.log_attendance_with_images(
        barcode, name, color, timestamp, full_frame_path, card_image_path, face_image_paths
    )

def save_scan_images(barcode: str, timestamp: datetime, 
                    full_image: bytes, card_image: Optional[bytes] = None,
                    face_images: Optional[List[bytes]] = None) -> bool:
    """Save scan images to the database (backward compatibility)."""
    return db_manager.save_scan_images(barcode, timestamp, full_image, card_image, face_images)

def fetch_attendance(date_filter: Optional[str] = None, 
                    color_filter: Optional[str] = None,
                    limit: Optional[int] = None,
                    offset: Optional[int] = 0) -> List[Dict[str, Any]]:
    """Fetch attendance records (backward compatibility)."""
    return db_manager.fetch_attendance(date_filter, color_filter, limit, offset)

def get_attendance_stats(days: int = 7) -> Dict[str, Any]:
    """Get attendance statistics (backward compatibility)."""
    return db_manager.get_attendance_stats(days)

def get_scan_images(barcode: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get scan images for a specific barcode (backward compatibility)."""
    return db_manager.get_scan_images(barcode, limit)

def get_image_as_base64(image_path: str) -> Optional[str]:
    """Get an image as a base64-encoded string (backward compatibility)."""
    return db_manager.get_image_as_base64(image_path)