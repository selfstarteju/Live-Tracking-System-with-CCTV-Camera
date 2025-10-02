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
        # Simple demo users (replace with real auth in production)
        demo_users = {
            'admin': {'password': 'admin123', 'role': 'admin'},
            'student1': {'password': 'student123', 'role': 'student', 'barcode': 'ID1001', 'name': 'Alice Demo'},
            'parent1': {'password': 'parent123', 'role': 'parent', 'child_barcode': 'ID1001', 'child_name': 'Alice Demo'}
        }

        user = demo_users.get(username)
        if user and user.get('password') == password:
            # Set session
            session['user_id'] = username
            session['username'] = username
            session['role'] = user.get('role')
            # Additional fields for student/parent
            if user.get('role') == 'student':
                session['barcode'] = user.get('barcode')
                session['fullname'] = user.get('name')
            if user.get('role') == 'parent':
                session['child_barcode'] = user.get('child_barcode')
                session['child_name'] = user.get('child_name')

            flash('Login successful!', 'success')
            # Redirect based on role
            if session['role'] == 'admin':
                return redirect(url_for('dashboard'))
            if session['role'] == 'student':
                return redirect(url_for('student_dashboard'))
            if session['role'] == 'parent':
                return redirect(url_for('parent_dashboard'))

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


@app.route('/api/presence', methods=['GET'])
@login_required
def api_presence():
    """Return presence for a barcode or all recent present users.

    Query params:
      - barcode: optional, returns presence for this barcode
      - within: seconds (default 300)
    """
    barcode = request.args.get('barcode')
    within = request.args.get('within', 300, type=int)

    if barcode:
        rec = db.get_latest_presence(barcode, within_seconds=within)
        return jsonify({'barcode': barcode, 'present': rec is not None, 'record': rec})

    # No barcode: return a list of recently-present barcodes
    # We'll query the attendance table for records within the time window
    cutoff = datetime.now() - timedelta(seconds=within)
    with_records = db.fetch_attendance()
    # Filter in Python to avoid broad SQL changes
    recent = [r for r in with_records if datetime.fromisoformat(r['timestamp']) >= cutoff]
    # Reduce to unique barcodes keeping latest
    latest = {}
    for r in sorted(recent, key=lambda x: x['timestamp'], reverse=True):
        if r['barcode'] not in latest:
            latest[r['barcode']] = r

    return jsonify({'present': list(latest.values())})


@app.route('/api/presence', methods=['POST'])
def api_presence_post():
    """Accept a pushed presence record from the tracker.

    Expects JSON: { barcode, name?, color?, verified?, timestamp? }
    """
    data = request.get_json() or {}
    barcode = data.get('barcode')
    if not barcode:
        return jsonify({'error': 'barcode required'}), 400

    name = data.get('name', '')
    color = data.get('color')
    verified = int(bool(data.get('verified', False)))
    ts = data.get('timestamp')
    try:
        timestamp = datetime.fromisoformat(ts) if ts else datetime.now()
    except Exception:
        timestamp = datetime.now()

    # Save attendance (no images in this push)
    ok = db.log_attendance_with_images(barcode, name, color, timestamp, None, None, None, verified=verified)
    if ok:
        return jsonify({'ok': True}), 200
    return jsonify({'ok': False}), 500


@app.route('/student')
@login_required
def student_dashboard():
    # Only students allowed
    if session.get('role') != 'student':
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    barcode = session.get('barcode')
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Fetch attendance for this student for the date (filtered in SQL)
    records = db.fetch_attendance(date_filter=date, color_filter='all', barcode=barcode)

    # Build chart data (counts per hour)
    hours = [f"{h}:00" for h in range(24)]
    counts = [0] * 24
    for r in records:
        try:
            ts = datetime.fromisoformat(r['timestamp'])
            counts[ts.hour] += 1
        except Exception:
            continue

    # Build multi-day trend (last 7 days)
    days = int(request.args.get('days', 7))
    trend_dates = []
    trend_counts = []
    today = datetime.strptime(date, '%Y-%m-%d')
    for i in range(days-1, -1, -1):
        d = today - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        trend_dates.append(d_str)
        ctr = len(db.fetch_attendance(date_filter=d_str, barcode=barcode))
        trend_counts.append(ctr)

    return render_template('student.html', records=records, date=date, hours=hours, counts=counts, fullname=session.get('fullname'), trend_dates=trend_dates, trend_counts=trend_counts)


@app.route('/parent')
@login_required
def parent_dashboard():
    if session.get('role') != 'parent':
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    child_barcode = session.get('child_barcode')
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    records = db.fetch_attendance(date_filter=date, color_filter='all', barcode=child_barcode)

    # Build chart data (counts per hour)
    hours = [f"{h}:00" for h in range(24)]
    counts = [0] * 24
    for r in records:
        try:
            ts = datetime.fromisoformat(r['timestamp'])
            counts[ts.hour] += 1
        except Exception:
            continue

    # Multi-day trend
    days = int(request.args.get('days', 7))
    trend_dates = []
    trend_counts = []
    today = datetime.strptime(date, '%Y-%m-%d')
    for i in range(days-1, -1, -1):
        d = today - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        trend_dates.append(d_str)
        ctr = len(db.fetch_attendance(date_filter=d_str, barcode=child_barcode))
        trend_counts.append(ctr)

    return render_template('parent.html', records=records, date=date, hours=hours, counts=counts, child_name=session.get('child_name'), trend_dates=trend_dates, trend_counts=trend_counts)


@app.route('/student_view/<barcode>')
@login_required
def student_view(barcode):
    """Admin/any logged-in user view for a student's attendance page by barcode."""
    # Accept date param
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Fetch attendance for this barcode and date
    records = db.fetch_attendance(date_filter=date, color_filter='all', barcode=barcode)

    # Hourly counts
    hours = [f"{h}:00" for h in range(24)]
    counts = [0] * 24
    for r in records:
        try:
            ts = datetime.fromisoformat(r['timestamp'])
            counts[ts.hour] += 1
        except Exception:
            continue

    # Multi-day trend (default last 7 days)
    days = int(request.args.get('days', 7))
    trend_dates = []
    trend_counts = []
    today = datetime.strptime(date, '%Y-%m-%d')
    for i in range(days-1, -1, -1):
        d = today - timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        trend_dates.append(d_str)
        ctr = len(db.fetch_attendance(date_filter=d_str, barcode=barcode))
        trend_counts.append(ctr)

    # Try to pick a name from latest record
    fullname = None
    if records:
        fullname = records[0].get('name') or barcode

    return render_template('student.html', records=records, date=date, hours=hours, counts=counts, fullname=fullname, trend_dates=trend_dates, trend_counts=trend_counts)


@app.route('/export_csv')
@login_required
def export_csv():
    # Accept barcode and date (or date range)
    barcode = request.args.get('barcode')
    date = request.args.get('date')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # build records list
    records = []
    if date:
        records = db.fetch_attendance(date_filter=date, barcode=barcode)
    elif start_date and end_date:
        # gather day by day
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            flash('Invalid date format for export', 'danger')
            return redirect(url_for('dashboard'))

        cur = start
        while cur <= end:
            dstr = cur.strftime('%Y-%m-%d')
            records.extend(db.fetch_attendance(date_filter=dstr, barcode=barcode))
            cur += timedelta(days=1)
    else:
        # default: today's date
        records = db.fetch_attendance(date_filter=datetime.now().strftime('%Y-%m-%d'), barcode=barcode)

    # Build CSV
    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'barcode', 'name', 'color', 'timestamp'])
    for r in records:
        writer.writerow([r.get('id'), r.get('barcode'), r.get('name'), r.get('color'), r.get('timestamp')])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'attendance_{barcode or "all"}_{date or (start_date+"_to_"+end_date if start_date and end_date else "today")}.csv'
    )

if __name__ == "__main__":
    db.init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)