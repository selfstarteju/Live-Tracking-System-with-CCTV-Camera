from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from datetime import datetime, timedelta
import database as db
import os
from functools import wraps
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Set a secret key for session management and flash messages
app.secret_key = os.environ.get('SECRET_KEY', 'your_secret_key_here')

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('You need to be logged in to access this page.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Simple authentication (in a real app, use a proper user database)
        if username == 'admin' and password == 'admin123':
            session['user_id'] = '1'
            session['username'] = 'admin'
            session['role'] = 'admin'
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get filter parameters
    date_filter = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    color_filter = request.args.get('color', 'all')
    
    # date_filter is already a string from the request, no need to convert
    # Just validate it's in the correct format
    try:
        # Validate the date string format
        datetime.strptime(date_filter, '%Y-%m-%d')
        filter_date = date_filter  # Use the string directly
    except ValueError:
        # If invalid format, use today's date
        filter_date = datetime.now().strftime('%Y-%m-%d')
        flash('Invalid date format. Using today\'s date.', 'warning')
    
    # Get records
    records = db.fetch_attendance(
        date_filter=filter_date,
        color_filter=color_filter
    )
    
    # Get statistics
    stats = db.get_attendance_stats(days=1)
    
    return render_template(
        'dashboard.html', 
        records=records, 
        date_filter=filter_date,
        color_filter=color_filter,
        total_records=stats['total_records'],
        color_counts=stats['by_color']
    )

@app.route('/api/records')
@login_required
def api_records():
    # Get filter parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    color = request.args.get('color', 'all')
    
    # Get records
    records = db.fetch_attendance(color_filter=color)
    
    # Filter by date range if provided
    if start_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            records = [r for r in records if r['timestamp'] >= start.isoformat()]
        except ValueError:
            return jsonify({'error': 'Invalid start_date format'}), 400
    
    if end_date:
        try:
            end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            records = [r for r in records if r['timestamp'] < end.isoformat()]
        except ValueError:
            return jsonify({'error': 'Invalid end_date format'}), 400
    
    return jsonify({
        'records': records,
        'count': len(records)
    })

@app.route('/api/stats')
@login_required
def api_stats():
    # Get time period
    days = request.args.get('days', 7, type=int)
    
    # Get statistics
    stats = db.get_attendance_stats(days=days)
    
    return jsonify(stats)

@app.route('/scan_images/<barcode>')
@login_required
def scan_images(barcode):
    """Display scan images for a specific barcode."""
    scan_records = db.get_scan_images(barcode)
    
    # Convert image paths to base64 for display
    for record in scan_records:
        if record.get('full_frame_path'):
            record['full_frame_base64'] = db.get_image_as_base64(record['full_frame_path'])
        
        if record.get('card_image_path'):
            record['card_image_base64'] = db.get_image_as_base64(record['card_image_path'])
        
        if record.get('face_image_paths'):
            record['face_images_base64'] = [
                db.get_image_as_base64(path) for path in record['face_image_paths']
            ]
    
    return render_template('scan_images.html', barcode=barcode, scan_records=scan_records)

@app.route('/image/<path:image_path>')
@login_required
def serve_image(image_path):
    """Serve an image file."""
    return send_file(image_path)

if __name__ == "__main__":
    db.init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)