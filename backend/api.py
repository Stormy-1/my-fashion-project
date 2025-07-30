from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import json
import cv2
import numpy as np
import base64
from werkzeug.utils import secure_filename
import traceback
import gc  # For garbage collection

# Lazy import to reduce memory usage at startup
def get_prediction_module():
    """Lazy import of prediction module to reduce memory usage"""
    try:
        from predict import process_image_for_recommendations
        return process_image_for_recommendations
    except ImportError as e:
        print(f"Warning: Could not import prediction module: {e}")
        return None

app = Flask(__name__)

# Configure CORS to allow both localhost and deployed frontend
CORS(app, origins=[
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # React dev server
    "http://127.0.0.1:5173", # Alternative localhost
    "https://fashion-recommendation-system-3.onrender.com",  # Your deployed frontend
    "https://*.onrender.com"  # Any Render subdomain
], supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Fashion Recommendation API is running'
    })

@app.route('/api/recommend', methods=['POST'])
def get_fashion_recommendations():
    """
    Main endpoint for fashion recommendations
    Expects:
    - image file (multipart/form-data)
    - height (form field, in cm)
    - weight (form field, in kg)
    - occasion (form field, optional, defaults to 'casual')
    
    Returns:
    - JSON response with recommendations and user info
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'error': 'No image file provided',
                'success': False
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'error': 'No image file selected',
                'success': False
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'success': False
            }), 400
        
        # Get form parameters
        try:
            height = float(request.form.get('height'))
            weight = float(request.form.get('weight'))
        except (TypeError, ValueError):
            return jsonify({
                'error': 'Height and weight must be valid numbers',
                'success': False
            }), 400
        
        # Validate height and weight ranges
        if not (100 <= height <= 250):  # cm
            return jsonify({
                'error': 'Height must be between 100 and 250 cm',
                'success': False
            }), 400
        
        if not (30 <= weight <= 300):  # kg
            return jsonify({
                'error': 'Weight must be between 30 and 300 kg',
                'success': False
            }), 400
        
        occasion = request.form.get('occasion', 'casual').lower()
        
        # Save uploaded file with unique name
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        print(f"Processing image: {filepath}")
        print(f"Parameters - Height: {height}cm, Weight: {weight}kg, Occasion: {occasion}")
        
        # Process the image using lazy-loaded pipeline
        process_func = get_prediction_module()
        if process_func is None:
            return jsonify({
                'error': 'Prediction module not available',
                'success': False
            }), 500
            
        recommendations = process_func(
            image_path=filepath,
            height=height,
            weight=weight,
            occasion=occasion
        )
        
        # Force garbage collection to free memory
        gc.collect()
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            print(f"Warning: Could not remove uploaded file {filepath}: {e}")
        
        # Load additional data if available
        facial_features = {}
        llm_recommendations = []
        
        try:
            if os.path.exists('facial_features.json'):
                with open('facial_features.json', 'r') as f:
                    facial_features = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load facial features: {e}")
        
        try:
            if os.path.exists('llm_recommendations.json'):
                with open('llm_recommendations.json', 'r') as f:
                    llm_recommendations = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load LLM recommendations: {e}")
        
        # Prepare response
        response_data = {
            'success': True,
            'user_info': {
                'height': height,
                'weight': weight,
                'occasion': occasion,
                'bmi': round(weight / ((height/100) ** 2), 2)
            },
            'facial_features': facial_features,
            'llm_recommendations': llm_recommendations,
            'scraped_products': recommendations if recommendations else [],
            'total_products': len(recommendations) if recommendations else 0
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_fashion_recommendations: {e}")
        traceback.print_exc()
        
        # Clean up uploaded file in case of error
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/camera-capture', methods=['POST'])
def camera_capture():
    """OpenCV camera capture endpoint for real-time photo capture and processing"""
    try:
        # Get form parameters
        try:
            height = float(request.form.get('height', 170))
            weight = float(request.form.get('weight', 70))
        except (TypeError, ValueError):
            return jsonify({
                'error': 'Height and weight must be valid numbers',
                'success': False
            }), 400
        
        # Validate height and weight ranges
        if not (100 <= height <= 250):  # cm
            return jsonify({
                'error': 'Height must be between 100 and 250 cm',
                'success': False
            }), 400
        
        if not (30 <= weight <= 300):  # kg
            return jsonify({
                'error': 'Weight must be between 30 and 300 kg',
                'success': False
            }), 400
        
        occasion = request.form.get('occasion', 'casual').lower()
        
        print(f"Starting OpenCV camera capture...")
        print(f"Parameters - Height: {height}cm, Weight: {weight}kg, Occasion: {occasion}")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({
                'error': 'Could not access camera. Please check if camera is available.',
                'success': False
            }), 500
        
        # Set camera properties for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized. Press SPACE to capture photo, ESC to cancel...")
        
        captured_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return jsonify({
                    'error': 'Failed to read from camera',
                    'success': False
                }), 500
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Add instructions overlay
            cv2.putText(frame, 'Press SPACE to capture, ESC to cancel', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Height: {height}cm, Weight: {weight}kg', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f'Occasion: {occasion}', 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Fashion Recommendation Camera - Press SPACE to capture', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Space key
                captured_frame = frame.copy()
                print("Photo captured!")
                break
            elif key == 27:  # Escape key
                cap.release()
                cv2.destroyAllWindows()
                return jsonify({
                    'error': 'Camera capture cancelled by user',
                    'success': False
                }), 400
        
        # Clean up camera resources
        cap.release()
        cv2.destroyAllWindows()
        
        if captured_frame is None:
            return jsonify({
                'error': 'No photo was captured',
                'success': False
            }), 400
        
        # Save captured frame temporarily
        temp_filename = f"camera_capture_{uuid.uuid4()}.jpg"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Save the captured frame
        cv2.imwrite(temp_filepath, captured_frame)
        print(f"Captured image saved to: {temp_filepath}")
        
        # Encode captured image as base64 for frontend display
        captured_image_base64 = None
        try:
            # Encode the captured frame directly to base64
            _, buffer = cv2.imencode('.jpg', captured_frame)
            captured_image_base64 = base64.b64encode(buffer).decode('utf-8')
            print("Captured image encoded as base64 for frontend")
        except Exception as e:
            print(f"Warning: Could not encode captured image as base64: {e}")
        
        # Process the captured image using the existing pipeline
        recommendations = process_image_for_recommendations(
            image_path=temp_filepath,
            height=height,
            weight=weight,
            occasion=occasion
        )
        
        # Clean up temporary file
        try:
            os.remove(temp_filepath)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {temp_filepath}: {e}")
        
        # Load additional data if available
        facial_features = {}
        llm_recommendations = []
        
        try:
            if os.path.exists('facial_features.json'):
                with open('facial_features.json', 'r') as f:
                    facial_features = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load facial features: {e}")
        
        try:
            if os.path.exists('llm_recommendations.json'):
                with open('llm_recommendations.json', 'r') as f:
                    llm_recommendations = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load LLM recommendations: {e}")
        
        # Prepare response
        response_data = {
            'success': True,
            'message': 'Photo captured and processed successfully',
            'user_info': {
                'height': height,
                'weight': weight,
                'occasion': occasion,
                'bmi': round(weight / ((height/100) ** 2), 2)
            },
            'captured_image': captured_image_base64,  # Add captured image as base64
            'facial_features': facial_features,
            'llm_recommendations': llm_recommendations,
            'scraped_products': recommendations if recommendations else [],
            'total_products': len(recommendations) if recommendations else 0
        }
        
        print(f"Camera capture completed successfully. Found {len(recommendations) if recommendations else 0} products.")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in camera_capture: {e}")
        traceback.print_exc()
        
        # Clean up in case of error
        try:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        except:
            pass
        
        return jsonify({
            'error': f'Camera capture error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/upload-test', methods=['POST'])
def test_upload():
    """Test endpoint for file upload functionality"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        return jsonify({
            'success': True,
            'message': 'File upload test successful',
            'filename': file.filename,
            'size': len(file.read())
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'File too large. Maximum size is 16MB.',
        'success': False
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500

if __name__ == '__main__':
    # Get port from environment variable (for Render deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting Fashion Recommendation API...")
    print(f"Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/recommend - Get fashion recommendations")
    print("  POST /api/camera-capture - OpenCV camera capture and processing")
    print("  POST /api/upload-test - Test file upload")
    print(f"\nServer starting on port {port}")
    
    # Use environment PORT for deployment compatibility
    app.run(debug=False, host='0.0.0.0', port=port)
