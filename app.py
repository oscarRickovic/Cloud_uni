from flask import Flask, render_template_string, request, jsonify, send_from_directory, session, redirect, url_for
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from functools import wraps
import redis
import os
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True
    )
    # Test the connection
    redis_client.ping()
    print("✅ Redis connection established")
except:
    print("❌ Redis connection failed. Using fallback authentication.")
    redis_client = None


def get_users():
    """Get users from Redis or fallback to hardcoded users"""
    if redis_client:
        try:
            # Get all users from Redis hash
            users = redis_client.hgetall("users")
            if users:
                return users
            else:
                # Initialize default users in Redis if empty
                default_users = {
                    'admin': 'password123',
                    'user1': 'demo123',
                    'researcher': 'umons2024',
                    'student': 'multimedia'
                }
                redis_client.hset("users", mapping=default_users)
                print("✅ Default users created in Redis")
                return default_users
        except Exception as e:
            print(f"❌ Redis error: {e}")
    
    # Fallback to hardcoded users
    return {
        'admin': 'password123',
        'user1': 'demo123',
        'researcher': 'umons2024',
        'student': 'multimedia'
    }
app = Flask(__name__)
app.secret_key = 'abdelhadi_agr'

# Ensure required directories exist
os.makedirs('uploads', exist_ok=True)


# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Similarity functions
def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors"""
    return float(np.sqrt(np.sum((vec1 - vec2) ** 2)))  # Convert to Python float

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))  # Convert to Python float

def chi_square_distance(vec1, vec2):
    """Calculate Chi-square distance"""
    n = min(len(vec1), len(vec2))
    epsilon = 1e-10  # Small value to avoid division by zero
    result = np.sum((vec1[:n] - vec2[:n])**2 / (vec2[:n] + epsilon))
    return float(result)  # Convert to Python float

def bhattacharyya_distance(vec1, vec2):
    """Calculate Bhattacharyya distance"""
    n = min(len(vec1), len(vec2))
    # Normalize vectors to make them probability distributions
    vec1_norm = vec1[:n] / (np.sum(vec1[:n]) + 1e-10)
    vec2_norm = vec2[:n] / (np.sum(vec2[:n]) + 1e-10)
    
    # Calculate Bhattacharyya coefficient
    bc = np.sum(np.sqrt(vec1_norm * vec2_norm))
    # Bhattacharyya distance
    result = -np.log(bc + 1e-10)
    return float(result)  # Convert to Python float

# Simple Image Searcher class
class SimpleImageSearcher:
    def __init__(self, features_folder="features", image_folder="image.orig"):
        self.features_folder = features_folder
        self.image_folder = image_folder
        self.models = {}
        self.image_dict = {}
        
        self.load_models()
        self.load_image_paths()
    
    def load_models(self):
        """Load all available model features from .pkl files"""
        model_files = {
            'VGG16': 'VGG16.pkl',
            'Resnet50': 'Resnet50.pkl', 
            'MobileNet': 'MobileNet.pkl'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(self.features_folder, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        features_data = pickle.load(f)
                    
                    if isinstance(features_data, list):
                        # Handle list of tuples (image_path, feature_vector)
                        features_dict = {}
                        for item in features_data:
                            if isinstance(item, tuple) and len(item) == 2:
                                image_path, feature_vector = item
                                # Extract image name from path
                                image_name = os.path.splitext(os.path.basename(image_path))[0]
                                # Ensure feature vector is a numpy array and flatten if needed
                                if isinstance(feature_vector, np.ndarray):
                                    feature_array = feature_vector.flatten().astype(np.float32)
                                else:
                                    feature_array = np.array(feature_vector).flatten().astype(np.float32)
                                features_dict[image_name] = feature_array
                        
                        if len(features_dict) > 0:
                            self.models[model_name] = features_dict
                            print(f"✅ Loaded {model_name} with {len(features_dict)} features")
                    
                    elif isinstance(features_data, dict):
                        # Convert numpy arrays to ensure consistency
                        processed_dict = {}
                        for key, value in features_data.items():
                            if isinstance(value, np.ndarray):
                                processed_dict[key] = value.flatten().astype(np.float32)
                            else:
                                processed_dict[key] = np.array(value).flatten().astype(np.float32)
                        
                        self.models[model_name] = processed_dict
                        print(f"✅ Loaded {model_name} (dict) with {len(processed_dict)} features")
                        
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
            else:
                print(f"⚠️  {filename} not found in {self.features_folder}")
    
    def load_image_paths(self):
        """Create a dictionary mapping image names to their paths"""
        if os.path.exists(self.image_folder):
            for filename in os.listdir(self.image_folder):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    base_name = os.path.splitext(filename)[0]
                    self.image_dict[base_name] = os.path.join(self.image_folder, filename)
            print(f"✅ Found {len(self.image_dict)} images in {self.image_folder}")
        else:
            print(f"⚠️  Image folder {self.image_folder} not found")
    
    def find_query_key(self, features_dict, query_name):
        """Find the actual key in features_dict that corresponds to query_name"""
        if query_name in features_dict:
            return query_name
        
        # Try with common extensions
        extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        for ext in extensions:
            if f"{query_name}{ext}" in features_dict:
                return f"{query_name}{ext}"
            if f"{query_name}{ext.upper()}" in features_dict:
                return f"{query_name}{ext.upper()}"
        
        # Try without extension if query_name has one
        if '.' in query_name:
            base_name = os.path.splitext(query_name)[0]
            if base_name in features_dict:
                return base_name
        
        return None
    
    def calculate_similarity(self, vec1, vec2, metric):
        """Calculate similarity based on the specified metric"""
        if metric == "euclidean":
            return euclidean_distance(vec1, vec2)
        elif metric == "cosine":
            return cosine_similarity(vec1, vec2)
        elif metric == "chi_square":
            return chi_square_distance(vec1, vec2)
        elif metric == "bhattacharyya":
            return bhattacharyya_distance(vec1, vec2)
        else:
            return euclidean_distance(vec1, vec2)  # Default to euclidean
    
    def get_k_neighbors(self, model_name, query_name, k=10, metric="euclidean"):
        """Find k nearest neighbors for a query image"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available models: {list(self.models.keys())}")
        
        features_dict = self.models[model_name]
        actual_key = self.find_query_key(features_dict, query_name)
        
        if actual_key is None:
            available_keys = list(features_dict.keys())[:10]
            raise ValueError(f"Query image '{query_name}' not found in {model_name} features! "
                           f"Available keys (first 10): {available_keys}")
        
        query_feature = features_dict[actual_key]
        results = []
        
        for name, feature_vector in features_dict.items():
            score = self.calculate_similarity(query_feature, feature_vector, metric)
            results.append((name, score))
        
        # Sort by distance (ascending for distance metrics, descending for similarity metrics)
        if metric == "cosine":
            results.sort(key=lambda x: x[1], reverse=True)  # Higher similarity = better
        else:
            results.sort(key=lambda x: x[1])  # Lower distance = better
        
        return results[:k]

# Initialize the search engine
print("🚀 Initializing search engine...")
try:
    searcher = SimpleImageSearcher(features_folder="features", image_folder="image.orig")
    AVAILABLE_MODELS = list(searcher.models.keys())
    print(f"📊 Available models: {AVAILABLE_MODELS}")
    print(f"📊 Model count: {len(AVAILABLE_MODELS)}")
except Exception as e:
    print(f"❌ Error initializing searcher: {e}")
    searcher = None
    AVAILABLE_MODELS = []

AVAILABLE_METRICS = ['euclidean', 'cosine', 'chi_square', 'bhattacharyya']

def calculate_precision_recall(query_name, results, total_relevant=20):
    """Calculate precision and recall for the results"""
    precision_recall_data = []
    
    # Determine the ground truth class (first digit of image name)
    try:
        query_class = int(query_name) // 100 if query_name.isdigit() else 0
    except:
        query_class = 0
    
    relevant_found = 0
    for i, (image_name, score) in enumerate(results):
        # Check if this result is relevant (same class)
        try:
            result_class = int(image_name) // 100 if image_name.isdigit() else -1
        except:
            result_class = -1
        
        if result_class == query_class:
            relevant_found += 1
        
        precision = relevant_found / (i + 1) if (i + 1) > 0 else 0
        recall = relevant_found / total_relevant if total_relevant > 0 else 0
        
        precision_recall_data.append({
            'rank': i + 1,
            'precision': float(precision),  # Ensure Python float
            'recall': float(recall),        # Ensure Python float
            'relevant': bool(result_class == query_class)  # Ensure Python bool
        })
    
    return precision_recall_data

def create_pr_curve(pr_data, model_name):
    """Create precision-recall curve as base64 image"""
    try:
        plt.figure(figsize=(10, 8))
        
        precisions = [p['precision'] for p in pr_data]
        recalls = [p['recall'] for p in pr_data]
        
        plt.plot(recalls, precisions, 'b-', linewidth=3, label=f'{model_name} PR Curve')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        
        # Add some styling
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_linewidth(0.5)
        plt.gca().spines['bottom'].set_linewidth(0.5)
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    except Exception as e:
        print(f"Error creating PR curve for {model_name}: {e}")
        return None

# HTML Templates
def get_login_template():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Multimedia Search Engine</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        body {
            background: var(--gradient-bg);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .login-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 25px;
            padding: 50px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 500px;
            width: 100%;
            margin: 20px;
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .login-header h1 {
            color: var(--primary-color);
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .login-header p {
            color: #6c757d;
            font-size: 1.1rem;
        }
        
        .form-floating {
            margin-bottom: 20px;
        }
        
        .form-control {
            border-radius: 15px;
            border: 2px solid #e9ecef;
            padding: 20px 15px;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .form-control:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
        }
        
        .btn-login {
            background: linear-gradient(135deg, var(--secondary-color) 0%, #2980b9 100%);
            border: none;
            border-radius: 25px;
            padding: 15px 30px;
            font-size: 1.1rem;
            font-weight: 500;
            color: white;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .btn-login:hover {
            background: linear-gradient(135deg, #2980b9 0%, var(--secondary-color) 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
        }
        
        .demo-credentials {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            margin-top: 30px;
            border-left: 4px solid var(--secondary-color);
        }
        
        .demo-credentials h6 {
            color: var(--primary-color);
            margin-bottom: 15px;
        }
        
        .credential-item {
            background: white;
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 10px;
            border: 1px solid #e9ecef;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .credential-item:hover {
            background: #e3f2fd;
            border-color: var(--secondary-color);
        }
        
        .alert {
            border-radius: 15px;
            margin-bottom: 20px;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .login-container {
            animation: fadeInUp 0.8s ease;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="login-header">
            <h1><i class="fas fa-lock me-3"></i>Login</h1>
            <p>Access the Multimedia Search Engine</p>
        </div>
        
        {% if error %}
        <div class="alert alert-danger" role="alert">
            <i class="fas fa-exclamation-triangle me-2"></i>{{ error }}
        </div>
        {% endif %}
        
        <form method="POST" id="loginForm">
            <div class="form-floating">
                <input type="text" class="form-control" id="username" name="username" placeholder="Username" required>
                <label for="username"><i class="fas fa-user me-2"></i>Username</label>
            </div>
            
            <div class="form-floating">
                <input type="password" class="form-control" id="password" name="password" placeholder="Password" required>
                <label for="password"><i class="fas fa-key me-2"></i>Password</label>
            </div>
            
            <button type="submit" class="btn btn-login">
                <i class="fas fa-sign-in-alt me-2"></i>Login to Search Engine
            </button>
        </form>
        
        <div class="demo-credentials">
            <h6><i class="fas fa-info-circle me-2"></i>Demo Credentials</h6>
            <p class="small text-muted mb-3">Click on any credential below to auto-fill the form:</p>
            
            <div class="credential-item" onclick="fillCredentials('admin', 'password123')">
                <strong>Admin User:</strong> admin / password123
            </div>
            <div class="credential-item" onclick="fillCredentials('researcher', 'umons2024')">
                <strong>Researcher:</strong> researcher / umons2024
            </div>
            <div class="credential-item" onclick="fillCredentials('student', 'multimedia')">
                <strong>Student:</strong> student / multimedia
            </div>
            <div class="credential-item" onclick="fillCredentials('user1', 'demo123')">
                <strong>Demo User:</strong> user1 / demo123
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        function fillCredentials(username, password) {
            document.getElementById('username').value = username;
            document.getElementById('password').value = password;
            
            // Add visual feedback
            const inputs = document.querySelectorAll('.form-control');
            inputs.forEach(input => {
                input.style.background = '#e8f5e8';
                setTimeout(() => {
                    input.style.background = '';
                }, 500);
            });
        }

        // Add Enter key support
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.ctrlKey && !event.metaKey) {
                document.getElementById('loginForm').submit();
            }
        });

        // Add form validation
        document.getElementById('loginForm').addEventListener('submit', function(e) {
            const username = document.getElementById('username').value.trim();
            const password = document.getElementById('password').value;
            
            if (!username || !password) {
                e.preventDefault();
                showAlert('Please enter both username and password.', 'warning');
            }
        });

        function showAlert(message, type) {
            const existingAlerts = document.querySelectorAll('.alert-notification');
            existingAlerts.forEach(alert => alert.remove());
            
            const alert = document.createElement('div');
            alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-notification`;
            alert.innerHTML = `<i class="fas fa-exclamation-triangle me-2"></i>${message}`;
            
            const form = document.getElementById('loginForm');
            form.parentNode.insertBefore(alert, form);
            
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 5000);
        }

        // Welcome animation
        setTimeout(() => {
            document.querySelector('.login-header p').style.color = '#3498db';
        }, 1000);
    </script>
</body>
</html>
    """

def get_main_template():
    return """
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimedia Search Engine - UMONS</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --accent-color: #e74c3c;
            --success-color: #27ae60;
            --info-color: #17a2b8;
            --background-color: #ecf0f1;
            --gradient-bg: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        body {
            background: var(--gradient-bg);
            min-height: 100vh;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
        }
        
        /* Header */
        .header {
            background: rgba(44, 62, 80, 0.95);
            backdrop-filter: blur(10px);
            color: white;
            padding: 20px 0;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
        }
        
        .header .subtitle {
            text-align: center;
            margin-top: 5px;
            opacity: 0.9;
            font-size: 1.1rem;
        }
        
        /* User Info Bar */
        .user-info {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        .user-welcome {
            color: #ecf0f1;
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-logout {
            background: linear-gradient(135deg, var(--accent-color) 0%, #c0392b 100%);
            border: none;
            border-radius: 20px;
            padding: 8px 16px;
            color: white;
            font-size: 0.85rem;
            transition: all 0.3s ease;
        }
        
        .btn-logout:hover {
            background: linear-gradient(135deg, #c0392b 0%, var(--accent-color) 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(231, 76, 60, 0.3);
        }
        
        /* Main container */
        .main-container {
            padding: 40px 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Cards */
        .card {
            border: none;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            backdrop-filter: blur(10px);
            background: rgba(255,255,255,0.95);
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 50px rgba(0,0,0,0.2);
        }
        
        .card-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border-radius: 20px 20px 0 0 !important;
            padding: 20px 30px;
            border: none;
        }
        
        .card-body {
            padding: 30px;
        }
        
        /* Statistics Dashboard */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: scale(1.05);
        }
        
        .stat-card h3 {
            font-size: 3rem;
            margin: 0;
            font-weight: bold;
        }
        
        .stat-card p {
            margin: 10px 0 0 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        /* Search Controls */
        .search-section {
            background: rgba(255,255,255,0.98);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            margin-bottom: 40px;
        }
        
        .form-control, .form-select {
            border-radius: 12px;
            border: 2px solid #e9ecef;
            padding: 12px 15px;
            transition: all 0.3s ease;
            font-size: 1rem;
        }
        
        .form-control:focus, .form-select:focus {
            border-color: var(--secondary-color);
            box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25);
        }
        
        .btn {
            border-radius: 25px;
            padding: 12px 30px;
            font-weight: 500;
            transition: all 0.3s ease;
            border: none;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--secondary-color) 0%, #2980b9 100%);
            font-size: 1.1rem;
        }
        
        .btn-primary:hover {
            background: linear-gradient(135deg, #2980b9 0%, var(--secondary-color) 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
        }
        
        .btn-info {
            background: linear-gradient(135deg, var(--info-color) 0%, #138496 100%);
            color: white;
        }
        
        .btn-info:hover {
            background: linear-gradient(135deg, #138496 0%, var(--info-color) 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(23, 162, 184, 0.4);
        }
        
        .btn-success {
            background: linear-gradient(135deg, var(--success-color) 0%, #229954 100%);
        }
        
        .btn-warning {
            background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        }
        
        /* Model Selection */
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .model-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            border: 2px solid #e9ecef;
            cursor: pointer;
        }
        
        .model-card:hover {
            background: #e3f2fd;
            border-color: var(--secondary-color);
        }
        
        .model-card.selected {
            background: linear-gradient(135deg, var(--secondary-color) 0%, #2980b9 100%);
            color: white;
            border-color: var(--secondary-color);
        }
        
        /* Combination Options */
        .combination-section {
            background: rgba(23, 162, 184, 0.1);
            border-radius: 15px;
            padding: 25px;
            margin-top: 20px;
            border: 2px solid rgba(23, 162, 184, 0.2);
        }
        
        .weight-slider-container {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .form-range {
            background: transparent;
        }
        
        .form-range::-webkit-slider-thumb {
            background: var(--info-color);
        }
        
        .form-range::-moz-range-thumb {
            background: var(--info-color);
            border: none;
        }
        
        /* Results Section */
        .query-section {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .query-image {
            max-height: 300px;
            border-radius: 15px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            transition: transform 0.3s ease;
        }
        
        .query-image:hover {
            transform: scale(1.05);
        }
        
        /* Model Results Tabs */
        .nav-tabs {
            border: none;
            margin-bottom: 30px;
        }
        
        .nav-tabs .nav-link {
            border: none;
            border-radius: 25px;
            margin-right: 10px;
            padding: 12px 25px;
            background: #f8f9fa;
            color: #495057;
            transition: all 0.3s ease;
        }
        
        .nav-tabs .nav-link.active {
            background: linear-gradient(135deg, var(--secondary-color) 0%, #2980b9 100%);
            color: white;
        }
        
        /* Combined Results Tab Special Styling */
        .nav-tabs .nav-link[id="tab-combined-tab"] {
            background: linear-gradient(135deg, var(--info-color) 0%, #138496 100%);
            color: white;
            font-weight: bold;
        }
        
        .nav-tabs .nav-link[id="tab-combined-tab"]:not(.active) {
            background: rgba(23, 162, 184, 0.2);
            color: var(--info-color);
        }
        
        /* Results Grid */
        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .result-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        
        .result-image {
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 10px;
            cursor: pointer;
            transition: transform 0.3s ease;
        }
        
        .result-image:hover {
            transform: scale(1.05);
        }
        
        /* Precision-Recall Chart */
        .pr-chart {
            max-width: 100%;
            height: auto;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Metrics Card */
        .metrics-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
        }
        
        .metrics-card h6 {
            border-bottom: 2px solid rgba(255,255,255,0.3);
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        
        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            padding: 60px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            width: 60px;
            height: 60px;
            border: 6px solid #f3f3f3;
            border-top: 6px solid var(--secondary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* About Section */
        .about-section {
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            padding: 40px;
            margin-top: 40px;
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        
        .team-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
        }
        
        /* Footer */
        .footer {
            background: rgba(44, 62, 80, 0.95);
            color: white;
            padding: 40px 0;
            margin-top: 60px;
            text-align: center;
        }
        
        /* Animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .animate-fade-in {
            animation: fadeInUp 0.8s ease;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .header h1 {
                font-size: 2rem;
            }
            
            .main-container {
                padding: 20px 10px;
            }
            
            .search-section {
                padding: 20px;
            }
            
            .card-body {
                padding: 20px;
            }
            
            .user-info {
                position: relative;
                top: auto;
                right: auto;
                text-align: center;
                margin-top: 10px;
                justify-content: center;
            }
            
            .btn {
                margin-bottom: 10px;
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container position-relative">
            <div class="user-info">
                <div class="user-welcome">
                    <i class="fas fa-user-circle"></i>
                    <span>Welcome, <strong>{{ username }}</strong></span>
                </div>
                <a href="{{ url_for('logout') }}" class="btn btn-logout">
                    <i class="fas fa-sign-out-alt me-1"></i>Logout
                </a>
            </div>
            <h1><i class="fas fa-search me-3"></i>Multimedia Search Engine</h1>
            <p class="subtitle">Abdelhadi Agourzam, Mohammed El-Ismayily | UMONS 2024-2025</p>
        </div>
    </header>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Statistics Dashboard -->
        <div class="stats-grid animate-fade-in">
            <div class="stat-card">
                <h3 id="total-images">-</h3>
                <p><i class="fas fa-images me-2"></i>Total Images</p>
            </div>
            <div class="stat-card">
                <h3 id="total-models">-</h3>
                <p><i class="fas fa-brain me-2"></i>AI Models</p>
            </div>
            <div class="stat-card">
                <h3 id="total-metrics">4</h3>
                <p><i class="fas fa-calculator me-2"></i>Similarity Metrics</p>
            </div>
            <div class="stat-card">
                <h3 id="search-count">0</h3>
                <p><i class="fas fa-search me-2"></i>Searches Performed</p>
            </div>
        </div>

        <!-- Search Controls -->
        <div class="search-section animate-fade-in">
            <h2 class="text-center mb-4">
                <i class="fas fa-search me-2"></i>Image Search Interface
            </h2>
            
            <form id="search-form">
                <div class="row">
                    <!-- Query Selection -->
                    <div class="col-lg-3 col-md-6 mb-3">
                        <label for="query-select" class="form-label fw-bold">
                            <i class="fas fa-image me-2"></i>Select Query Image
                        </label>
                        <select class="form-select" id="query-select" required>
                            <option value="">Choose an image...</option>
                        </select>
                    </div>

                    <!-- Similarity Metric -->
                    <div class="col-lg-2 col-md-6 mb-3">
                        <label for="metric-select" class="form-label fw-bold">
                            <i class="fas fa-ruler me-2"></i>Similarity Metric
                        </label>
                        <select class="form-select" id="metric-select">
                            <option value="euclidean" selected>Euclidean Distance</option>
                            <option value="cosine">Cosine Similarity</option>
                            <option value="chi_square">Chi-Square Distance</option>
                            <option value="bhattacharyya">Bhattacharyya Distance</option>
                        </select>
                    </div>

                    <!-- Top-K Selection -->
                    <div class="col-lg-2 col-md-6 mb-3">
                        <label for="top-k" class="form-label fw-bold">
                            <i class="fas fa-list-ol me-2"></i>Results Count
                        </label>
                        <select class="form-select" id="top-k">
                            <option value="10">Top 10</option>
                            <option value="20" selected>Top 20</option>
                            <option value="50">Top 50</option>
                        </select>
                    </div>

                    <!-- Search Buttons -->
                    <div class="col-lg-5 col-md-6 mb-3 d-flex align-items-end flex-wrap">
                        <button type="submit" class="btn btn-primary btn-lg me-2 mb-2" id="search-individual">
                            <i class="fas fa-search me-2"></i>Search Individual Models
                        </button>
                        <button type="button" class="btn btn-info btn-lg me-2 mb-2" id="search-combined" style="display: none;">
                            <i class="fas fa-layer-group me-2"></i>Search Combined
                        </button>
                        <button type="button" class="btn btn-success btn-lg me-2 mb-2" onclick="loadRandomImage()">
                            <i class="fas fa-random me-2"></i>Random
                        </button>
                        <button type="button" class="btn btn-warning btn-lg mb-2" onclick="clearResults()">
                            <i class="fas fa-trash me-2"></i>Clear
                        </button>
                    </div>
                </div>

                <!-- Model Selection -->
                <div class="row mt-4">
                    <div class="col-12">
                        <label class="form-label fw-bold">
                            <i class="fas fa-brain me-2"></i>Select AI Models (Choose one or more)
                        </label>
                        <div class="model-grid" id="model-selection">
                            <!-- Models will be populated by JavaScript -->
                        </div>
                    </div>
                </div>

                <!-- Combination Options (Show when multiple models selected) -->
                <div class="row mt-4" id="combination-options" style="display: none;">
                    <div class="col-12">
                        <div class="combination-section">
                            <label class="form-label fw-bold">
                                <i class="fas fa-layer-group me-2"></i>Model Combination Settings
                            </label>
                            <p class="text-muted mb-3">Configure how multiple models should be combined for better results</p>
                            
                            <div class="row">
                                <div class="col-md-4 mb-3">
                                    <label for="combination-method" class="form-label fw-bold">Combination Method</label>
                                    <select class="form-select" id="combination-method">
                                        <option value="average" selected>Average Scores</option>
                                        <option value="weighted">Weighted Combination</option>
                                        <option value="rank_fusion">Rank Fusion</option>
                                    </select>
                                    <small class="text-muted">
                                        <div id="method-description">Combines scores using simple averaging</div>
                                    </small>
                                </div>
                                
                                <div class="col-md-8" id="weight-controls" style="display: none;">
                                    <label class="form-label fw-bold">Model Weights</label>
                                    <p class="small text-muted mb-2">Adjust the importance of each model (higher = more influence)</p>
                                    <div id="weight-sliders">
                                        <!-- Weight sliders will be populated by JavaScript -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </form>
        </div>

        <!-- Loading Indicator -->
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h4>Searching for similar images...</h4>
            <p class="text-muted">Processing your query with selected AI models</p>
        </div>

        <!-- Results Section -->
        <div id="results-section" style="display: none;">
            <!-- Query Image Display -->
            <div class="card animate-fade-in">
                <div class="card-header">
                    <h3><i class="fas fa-bullseye me-2"></i>Query Image</h3>
                </div>
                <div class="card-body query-section">
                    <img id="query-image" src="" alt="Query Image" class="query-image">
                    <h4 id="query-info" class="mt-3"></h4>
                </div>
            </div>

            <!-- Model Results -->
            <div class="card animate-fade-in">
                <div class="card-header">
                    <h3><i class="fas fa-chart-bar me-2"></i>Search Results by Model</h3>
                </div>
                <div class="card-body">
                    <!-- Navigation Tabs -->
                    <ul class="nav nav-tabs" id="model-tabs" role="tablist">
                        <!-- Tabs will be populated by JavaScript -->
                    </ul>

                    <!-- Tab Content -->
                    <div class="tab-content" id="model-tab-content">
                        <!-- Content will be populated by JavaScript -->
                    </div>
                </div>
            </div>
        </div>

        <!-- About Section -->
        <div class="about-section animate-fade-in">
            <h2 class="text-center mb-4">
                <i class="fas fa-info-circle me-2"></i>About This Project
            </h2>
            
            <div class="row">
                <div class="col-lg-8">
                    <h4><i class="fas fa-project-diagram me-2"></i>Project Overview</h4>
                    <p class="lead">
                        This multimedia search engine represents a comprehensive implementation of modern 
                        computer vision and cloud computing technologies for content-based image retrieval.
                    </p>
                    
                    <h5><i class="fas fa-bullseye me-2"></i>Key Features</h5>
                    <ul class="list-unstyled">
                        <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Multiple AI models (VGG16, MobileNet, ResNet50, Vision Transformers)</li>
                        <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Various similarity metrics for comprehensive comparison</li>
                        <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Model combination with weighted and rank fusion methods</li>
                        <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Real-time precision-recall analysis</li>
                        <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Cloud-based scalable architecture</li>
                        <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Secure user authentication system</li>
                        <li class="mb-2"><i class="fas fa-check text-success me-2"></i>Responsive web interface</li>
                    </ul>
                </div>
                
                <div class="col-lg-4">
                    <h4><i class="fas fa-layer-group me-2"></i>Technical Stack</h4>
                    <div class="mb-3">
                        <h6>Backend:</h6>
                        <span class="badge bg-primary me-1">Python</span>
                        <span class="badge bg-primary me-1">Flask</span>
                        <span class="badge bg-primary me-1">PyTorch</span>
                        <span class="badge bg-primary me-1">NumPy</span>
                    </div>
                    <div class="mb-3">
                        <h6>Frontend:</h6>
                        <span class="badge bg-info me-1">HTML5</span>
                        <span class="badge bg-info me-1">CSS3</span>
                        <span class="badge bg-info me-1">JavaScript</span>
                        <span class="badge bg-info me-1">Bootstrap 5</span>
                    </div>
                    <div class="mb-3">
                        <h6>Infrastructure:</h6>
                        <span class="badge bg-success me-1">Docker</span>
                        <span class="badge bg-success me-1">Cloud VM</span>
                        <span class="badge bg-success me-1">Session Auth</span>
                    </div>
                </div>
            </div>

            <!-- Team Information -->
            <div class="team-grid">
                <div class="team-card">
                    <i class="fas fa-graduation-cap fa-3x mb-3"></i>
                    <h4>University of Mons</h4>
                    <p>UMONS - Faculty of Engineering</p>
                    <h6>Course: Cloud & Edge Computing</h6>
                    <p>Academic Year 2024-2025</p>
                </div>
                <div class="team-card">
                    <i class="fas fa-users fa-3x mb-3"></i>
                    <h4>Supervisors</h4>
                    <p><strong>Prof. Sidi Ahmed Mahmoudi</strong></p>
                    <p><strong>Dr. Aurélie Cools</strong></p>
                    <p class="mt-3">Project Timeline: Feb - June 2025</p>
                </div>
            </div>
        </div>
    </div>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>&copy; 2024-2025 University of Mons (UMONS) - Cloud & Edge Computing Project</p>
            <p>Multimedia Search Engine using Deep Learning CNN and Vision Transformers</p>
            <p>Logged in as: <strong>{{ username }}</strong> | Session secured</p>
        </div>
    </footer>

    <!-- Scripts -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    
    <script>
        let searchCount = 0;
        let availableModels = [];
        let selectedModels = [];

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            loadAvailableImages();
            loadStats();
            document.getElementById('search-form').addEventListener('submit', handleSearch);
            
            // Combined search button event listener
            document.getElementById('search-combined').addEventListener('click', handleCombinedSearch);
            
            // Combination method change event listener
            document.getElementById('combination-method').addEventListener('change', function() {
                updateWeightControls();
                updateMethodDescription();
            });
            
            // Update combination UI initially
            updateCombinationUI();
            updateMethodDescription();
            
            // Welcome message with username
            setTimeout(() => {
                showAlert('Welcome back, {{ username }}! Ready to explore multimedia search.', 'success');
            }, 1000);
        });

        // Load available images
        async function loadAvailableImages() {
            try {
                const response = await fetch('/api/available_images');
                const data = await response.json();
                
                const select = document.getElementById('query-select');
                select.innerHTML = '<option value="">Choose an image...</option>';
                
                data.images.forEach(image => {
                    const option = document.createElement('option');
                    option.value = image;
                    option.textContent = `Image ${image}`;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading images:', error);
                showAlert('Error loading images. Please refresh the page.', 'error');
            }
        }

        // Load application statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('total-images').textContent = data.total_images;
                document.getElementById('total-models').textContent = data.available_models.length;
                
                // Populate model selection
                availableModels = data.available_models;
                selectedModels = [...availableModels]; // Select all by default
                populateModelSelection(data.available_models);
                
            } catch (error) {
                console.error('Error loading stats:', error);
                showAlert('Error loading application statistics.', 'error');
            }
        }

        // Populate model selection grid
        function populateModelSelection(models) {
            const container = document.getElementById('model-selection');
            container.innerHTML = '';
            
            models.forEach(model => {
                const modelCard = document.createElement('div');
                modelCard.className = 'model-card selected';
                modelCard.innerHTML = `
                    <i class="fas fa-brain fa-2x mb-2"></i>
                    <h6>${model}</h6>
                    <small>Deep Learning Model</small>
                `;
                
                modelCard.addEventListener('click', function() {
                    toggleModelSelection(model, modelCard);
                });
                
                container.appendChild(modelCard);
            });
            
            // Update combination UI after populating models
            updateCombinationUI();
        }

        // Toggle model selection
        function toggleModelSelection(model, element) {
            if (selectedModels.includes(model)) {
                selectedModels = selectedModels.filter(m => m !== model);
                element.classList.remove('selected');
            } else {
                selectedModels.push(model);
                element.classList.add('selected');
            }
            
            // Ensure at least one model is selected
            if (selectedModels.length === 0) {
                selectedModels.push(model);
                element.classList.add('selected');
                showAlert('At least one model must be selected.', 'warning');
            }
            
            // Show/hide combination options and combined search button
            updateCombinationUI();
        }

        // Update combination UI based on selected models
        function updateCombinationUI() {
            const combinationOptions = document.getElementById('combination-options');
            const combinedSearchBtn = document.getElementById('search-combined');
            const individualSearchBtn = document.getElementById('search-individual');
            
            if (selectedModels.length >= 2) {
                combinationOptions.style.display = 'block';
                combinedSearchBtn.style.display = 'inline-block';
                individualSearchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Search Individual Models';
                
                // Update weight controls
                updateWeightControls();
            } else {
                combinationOptions.style.display = 'none';
                combinedSearchBtn.style.display = 'none';
                individualSearchBtn.innerHTML = '<i class="fas fa-search me-2"></i>Search Similar Images';
            }
        }

        // Update weight controls for selected models
        function updateWeightControls() {
            const weightSliders = document.getElementById('weight-sliders');
            const combinationMethod = document.getElementById('combination-method').value;
            const weightControls = document.getElementById('weight-controls');
            
            if (combinationMethod === 'weighted') {
                weightControls.style.display = 'block';
                weightSliders.innerHTML = '';
                
                selectedModels.forEach(model => {
                    const sliderDiv = document.createElement('div');
                    sliderDiv.className = 'weight-slider-container';
                    sliderDiv.innerHTML = `
                        <label class="form-label small fw-bold">${model}</label>
                        <div class="d-flex align-items-center">
                            <input type="range" class="form-range me-3" 
                                   id="weight-${model}" min="0.1" max="2.0" step="0.1" value="1.0">
                            <span class="badge bg-info" id="weight-value-${model}">1.0</span>
                        </div>
                    `;
                    weightSliders.appendChild(sliderDiv);
                    
                    // Add event listener for real-time updates
                    const slider = document.getElementById(`weight-${model}`);
                    const valueDisplay = document.getElementById(`weight-value-${model}`);
                    slider.addEventListener('input', function() {
                        valueDisplay.textContent = this.value;
                    });
                });
            } else {
                weightControls.style.display = 'none';
            }
        }

        // Update method description
        function updateMethodDescription() {
            const method = document.getElementById('combination-method').value;
            const description = document.getElementById('method-description');
            
            const descriptions = {
                'average': 'Combines scores using simple averaging',
                'weighted': 'Allows custom weights for each model',
                'rank_fusion': 'Combines based on ranking positions'
            };
            
            description.textContent = descriptions[method] || '';
        }

        // Handle search form submission
        async function handleSearch(event) {
            event.preventDefault();
            
            const queryName = document.getElementById('query-select').value;
            const metric = document.getElementById('metric-select').value;
            const topK = document.getElementById('top-k').value;
            
            if (!queryName) {
                showAlert('Please select a query image.', 'warning');
                return;
            }
            
            if (selectedModels.length === 0) {
                showAlert('Please select at least one AI model.', 'warning');
                return;
            }
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('results-section').style.display = 'none';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query_name: queryName,
                        models: selectedModels,
                        metric: metric,
                        top_k: parseInt(topK)
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayResults(data);
                    searchCount++;
                    document.getElementById('search-count').textContent = searchCount;
                    showAlert('Search completed successfully!', 'success');
                } else {
                    if (response.status === 401) {
                        showAlert('Session expired. Redirecting to login...', 'error');
                        setTimeout(() => {
                            window.location.href = '/login';
                        }, 2000);
                    } else {
                        showAlert('Search failed: ' + data.error, 'error');
                    }
                }
            } catch (error) {
                console.error('Search error:', error);
                showAlert('Search failed. Please try again.', 'error');
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }

        // Handle combined search
        async function handleCombinedSearch() {
            const queryName = document.getElementById('query-select').value;
            const metric = document.getElementById('metric-select').value;
            const topK = document.getElementById('top-k').value;
            const combinationMethod = document.getElementById('combination-method').value;
            
            if (!queryName) {
                showAlert('Please select a query image.', 'warning');
                return;
            }
            
            if (selectedModels.length < 2) {
                showAlert('Please select at least 2 models for combination.', 'warning');
                return;
            }
            
            // Get model weights if using weighted combination
            const modelWeights = {};
            if (combinationMethod === 'weighted') {
                selectedModels.forEach(model => {
                    const weightSlider = document.getElementById(`weight-${model}`);
                    if (weightSlider) {
                        modelWeights[model] = parseFloat(weightSlider.value);
                    }
                });
            }
            
            // Show loading
            document.getElementById('loading').classList.add('show');
            document.getElementById('results-section').style.display = 'none';
            
            try {
                const response = await fetch('/api/search_combined', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query_name: queryName,
                        models: selectedModels,
                        metric: metric,
                        top_k: parseInt(topK),
                        combination_method: combinationMethod,
                        model_weights: modelWeights
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    displayCombinedResults(data);
                    searchCount++;
                    document.getElementById('search-count').textContent = searchCount;
                    showAlert(`Combined search completed using ${combinationMethod} method!`, 'success');
                } else {
                    if (response.status === 401) {
                        showAlert('Session expired. Redirecting to login...', 'error');
                        setTimeout(() => {
                            window.location.href = '/login';
                        }, 2000);
                    } else {
                        showAlert('Combined search failed: ' + data.error, 'error');
                    }
                }
            } catch (error) {
                console.error('Combined search error:', error);
                showAlert('Combined search failed. Please try again.', 'error');
            } finally {
                document.getElementById('loading').classList.remove('show');
            }
        }

        // Display search results
        function displayResults(data) {
            // Display query image
            const queryImg = document.getElementById('query-image');
            queryImg.src = `/images/${data.query_name}.jpg`;
            document.getElementById('query-info').textContent = `Query: Image ${data.query_name} | Metric: ${data.metric || 'euclidean'} | Time: ${new Date().toLocaleTimeString()}`;
            
            // Create tabs for models
            const tabsContainer = document.getElementById('model-tabs');
            const contentContainer = document.getElementById('model-tab-content');
            
            tabsContainer.innerHTML = '';
            contentContainer.innerHTML = '';
            
            let firstTab = true;
            
            Object.keys(data.results).forEach(modelName => {
                if (data.results[modelName].error) {
                    console.error(`Error in ${modelName}:`, data.results[modelName].error);
                    return;
                }
                
                // Create tab
                const tabId = `tab-${modelName.toLowerCase()}`;
                const tab = document.createElement('li');
                tab.className = 'nav-item';
                tab.innerHTML = `
                    <button class="nav-link ${firstTab ? 'active' : ''}" 
                            id="${tabId}-tab" data-bs-toggle="tab" 
                            data-bs-target="#${tabId}" type="button">
                        <i class="fas fa-brain me-2"></i>${modelName}
                    </button>
                `;
                tabsContainer.appendChild(tab);
                
                // Create tab content
                const tabContent = document.createElement('div');
                tabContent.className = `tab-pane fade ${firstTab ? 'show active' : ''}`;
                tabContent.id = tabId;
                
                // Results grid
                const resultsGrid = document.createElement('div');
                resultsGrid.className = 'results-grid';
                
                const results = data.results[modelName];
                results.forEach((result, index) => {
                    const [imageName, score] = result;
                    const resultCard = document.createElement('div');
                    resultCard.className = 'result-card';
                    resultCard.innerHTML = `
                        <img src="/images/${imageName}.jpg" alt="Result ${index + 1}" 
                             class="result-image" onclick="showImageModal('/images/${imageName}.jpg', 'Image ${imageName}')">
                        <h6 class="mt-2 mb-1">Rank ${index + 1}</h6>
                        <p class="mb-1"><strong>Image ${imageName}</strong></p>
                        <p class="text-muted small mb-0">Score: ${score.toFixed(4)}</p>
                    `;
                    resultsGrid.appendChild(resultCard);
                });
                
                // Precision-Recall curve section
                const prSection = document.createElement('div');
                prSection.className = 'mt-4 text-center';
                prSection.innerHTML = `
                    <h5><i class="fas fa-chart-line me-2"></i>Precision-Recall Analysis - ${modelName}</h5>
                    <div class="mt-3">
                        <img src="data:image/png;base64,${data.pr_curves[modelName]}" 
                             alt="Precision-Recall Curve" class="pr-chart">
                    </div>
                `;
                
                tabContent.appendChild(resultsGrid);
                tabContent.appendChild(prSection);
                contentContainer.appendChild(tabContent);
                
                firstTab = false;
            });
            
            // Show results section
            document.getElementById('results-section').style.display = 'block';
            document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
        }

        // Display combined search results
        function displayCombinedResults(data) {
            // Display query image
            const queryImg = document.getElementById('query-image');
            queryImg.src = `/images/${data.query_name}.jpg`;
            document.getElementById('query-info').textContent = 
                `Query: Image ${data.query_name} | Combined Method: ${data.combination_method} | Time: ${new Date().toLocaleTimeString()}`;
            
            // Create tabs for individual models + combined results
            const tabsContainer = document.getElementById('model-tabs');
            const contentContainer = document.getElementById('model-tab-content');
            
            tabsContainer.innerHTML = '';
            contentContainer.innerHTML = '';
            
            // Combined results tab (first)
            const combinedTab = document.createElement('li');
            combinedTab.className = 'nav-item';
            combinedTab.innerHTML = `
                <button class="nav-link active" 
                        id="tab-combined-tab" data-bs-toggle="tab" 
                        data-bs-target="#tab-combined" type="button">
                    <i class="fas fa-layer-group me-2"></i>Combined Results
                </button>
            `;
            tabsContainer.appendChild(combinedTab);
            
            // Combined results content
            const combinedContent = document.createElement('div');
            combinedContent.className = 'tab-pane fade show active';
            combinedContent.id = 'tab-combined';
            
            // Combined results grid
            const combinedGrid = document.createElement('div');
            combinedGrid.className = 'results-grid';
            
            data.combined_results.forEach((result, index) => {
                const [imageName, score] = result;
                const resultCard = document.createElement('div');
                resultCard.className = 'result-card';
                resultCard.innerHTML = `
                    <img src="/images/${imageName}.jpg" alt="Result ${index + 1}" 
                         class="result-image" onclick="showImageModal('/images/${imageName}.jpg', 'Image ${imageName}')">
                    <h6 class="mt-2 mb-1">Rank ${index + 1}</h6>
                    <p class="mb-1"><strong>Image ${imageName}</strong></p>
                    <p class="text-muted small mb-0">Score: ${score.toFixed(4)}</p>
                `;
                combinedGrid.appendChild(resultCard);
            });
            
            // Combined metrics and PR curve
            const combinedMetrics = document.createElement('div');
            combinedMetrics.className = 'mt-4';
            combinedMetrics.innerHTML = `
                <h5><i class="fas fa-chart-line me-2"></i>Combined Results Analysis</h5>
                <div class="row">
                    <div class="col-md-8 text-center">
                        <img src="data:image/png;base64,${data.pr_curve}" 
                             alt="Combined Precision-Recall Curve" class="pr-chart">
                    </div>
                    <div class="col-md-4">
                        <div class="metrics-card">
                            <h6><i class="fas fa-chart-bar me-2"></i>Performance Metrics</h6>
                            <p><strong>Method:</strong> ${data.combination_method}</p>
                            <p><strong>Models Used:</strong> ${selectedModels.join(', ')}</p>
                            <p><strong>Precision@${data.combined_results.length}:</strong> 
                               ${(data.advanced_metrics.precision_at_k * 100).toFixed(1)}%</p>
                            <p><strong>Recall@${data.combined_results.length}:</strong> 
                               ${(data.advanced_metrics.recall_at_k * 100).toFixed(1)}%</p>
                            <p><strong>Relevant Found:</strong> ${data.advanced_metrics.relevant_found}</p>
                            ${data.model_weights && Object.keys(data.model_weights).length > 0 ? 
                                `<hr><h6>Model Weights:</h6>` + 
                                Object.entries(data.model_weights).map(([model, weight]) => 
                                    `<p><strong>${model}:</strong> ${weight}</p>`
                                ).join('') : ''
                            }
                        </div>
                    </div>
                </div>
            `;
            
            combinedContent.appendChild(combinedGrid);
            combinedContent.appendChild(combinedMetrics);
            contentContainer.appendChild(combinedContent);
            
            // Individual model tabs
            Object.keys(data.individual_results).forEach(modelName => {
                const tabId = `tab-${modelName.toLowerCase()}`;
                const tab = document.createElement('li');
                tab.className = 'nav-item';
                tab.innerHTML = `
                    <button class="nav-link" 
                            id="${tabId}-tab" data-bs-toggle="tab" 
                            data-bs-target="#${tabId}" type="button">
                        <i class="fas fa-brain me-2"></i>${modelName}
                    </button>
                `;
                tabsContainer.appendChild(tab);
                
                // Individual model content
                const tabContent = document.createElement('div');
                tabContent.className = 'tab-pane fade';
                tabContent.id = tabId;
                
                const resultsGrid = document.createElement('div');
                resultsGrid.className = 'results-grid';
                
                const results = data.individual_results[modelName];
                results.forEach((result, index) => {
                    const [imageName, score] = result;
                    const resultCard = document.createElement('div');
                    resultCard.className = 'result-card';
                    resultCard.innerHTML = `
                        <img src="/images/${imageName}.jpg" alt="Result ${index + 1}" 
                             class="result-image" onclick="showImageModal('/images/${imageName}.jpg', 'Image ${imageName}')">
                        <h6 class="mt-2 mb-1">Rank ${index + 1}</h6>
                        <p class="mb-1"><strong>Image ${imageName}</strong></p>
                        <p class="text-muted small mb-0">Score: ${score.toFixed(4)}</p>
                    `;
                    resultsGrid.appendChild(resultCard);
                });
                
                tabContent.appendChild(resultsGrid);
                contentContainer.appendChild(tabContent);
            });
            
            // Show results section
            document.getElementById('results-section').style.display = 'block';
            document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
        }

        // Load random image
        function loadRandomImage() {
            const select = document.getElementById('query-select');
            const options = select.getElementsByTagName('option');
            
            if (options.length > 1) {
                const randomIndex = Math.floor(Math.random() * (options.length - 1)) + 1;
                select.selectedIndex = randomIndex;
                showAlert(`Random image selected: ${options[randomIndex].text}`, 'info');
            }
        }

        // Clear results
        function clearResults() {
            document.getElementById('results-section').style.display = 'none';
            document.getElementById('query-select').selectedIndex = 0;
            showAlert('Results cleared.', 'info');
        }

        // Show image modal
        function showImageModal(imageSrc, imageTitle) {
            const modal = document.createElement('div');
            modal.className = 'modal fade';
            modal.innerHTML = `
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="fas fa-image me-2"></i>${imageTitle}
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body text-center">
                            <img src="${imageSrc}" alt="${imageTitle}" class="img-fluid" style="max-height: 70vh;">
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            const modalInstance = new bootstrap.Modal(modal);
            modalInstance.show();
            
            modal.addEventListener('hidden.bs.modal', function() {
                document.body.removeChild(modal);
            });
        }

        // Show alert messages
        function showAlert(message, type) {
            // Remove existing alerts
            const existingAlerts = document.querySelectorAll('.alert-notification');
            existingAlerts.forEach(alert => alert.remove());
            
            const alert = document.createElement('div');
            alert.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show alert-notification`;
            alert.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 9999;
                min-width: 300px;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            `;
            
            alert.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : type === 'error' ? 'times-circle' : 'info-circle'} me-2"></i>
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
            
            document.body.appendChild(alert);
            
            // Auto remove after 5 seconds
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.remove();
                }
            }, 5000);
        }

        // Session timeout warning
        let sessionTimeout;
        function resetSessionTimeout() {
            clearTimeout(sessionTimeout);
            sessionTimeout = setTimeout(() => {
                showAlert('Your session will expire in 5 minutes. Please save your work.', 'warning');
            }, 25 * 60 * 1000); // 25 minutes warning for 30 minute session
        }

        // Reset timeout on user activity
        document.addEventListener('click', resetSessionTimeout);
        document.addEventListener('keypress', resetSessionTimeout);
        resetSessionTimeout(); // Initial call

        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            // Ctrl+Enter or Cmd+Enter to search
            if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                event.preventDefault();
                if (selectedModels.length >= 2 && document.getElementById('search-combined').style.display !== 'none') {
                    handleCombinedSearch();
                } else {
                    document.getElementById('search-form').dispatchEvent(new Event('submit'));
                }
            }
            
            // Escape to clear results
            if (event.key === 'Escape') {
                clearResults();
            }
            
            // R key for random image
            if (event.key.toLowerCase() === 'r' && !event.ctrlKey && !event.metaKey) {
                if (document.activeElement.tagName !== 'INPUT' && document.activeElement.tagName !== 'SELECT') {
                    loadRandomImage();
                }
            }
            
            // Ctrl+L for logout
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'l') {
                event.preventDefault();
                if (confirm('Are you sure you want to logout?')) {
                    window.location.href = '/logout';
                }
            }
        });

        // Add some interactive feedback
        document.addEventListener('click', function(e) {
            if (e.target.classList.contains('btn')) {
                e.target.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    e.target.style.transform = '';
                }, 100);
            }
        });

        // Logout confirmation
        document.querySelector('.btn-logout').addEventListener('click', function(e) {
            if (!confirm('Are you sure you want to logout?')) {
                e.preventDefault();
            }
        });

        // Add tooltips to buttons
        document.addEventListener('DOMContentLoaded', function() {
            // Add title attributes for tooltips
            setTimeout(() => {
                const randomBtn = document.querySelector('button[onclick="loadRandomImage()"]');
                const clearBtn = document.querySelector('button[onclick="clearResults()"]');
                const searchBtn = document.querySelector('button[type="submit"]');
                const combinedBtn = document.getElementById('search-combined');
                const logoutBtn = document.querySelector('.btn-logout');
                
                if (randomBtn) randomBtn.title = 'Select a random image (Press R)';
                if (clearBtn) clearBtn.title = 'Clear search results (Press Escape)';
                if (searchBtn) searchBtn.title = 'Start individual model search (Ctrl+Enter)';
                if (combinedBtn) combinedBtn.title = 'Start combined model search (Ctrl+Enter when multiple models selected)';
                if (logoutBtn) logoutBtn.title = 'Logout from the system (Ctrl+L)';
            }, 1000);
        });

        // Performance monitoring
        let searchStartTime;
        
        const originalHandleSearch = handleSearch;
        handleSearch = async function(event) {
            searchStartTime = performance.now();
            await originalHandleSearch(event);
            const searchTime = ((performance.now() - searchStartTime) / 1000).toFixed(2);
            console.log(`Individual search completed in ${searchTime} seconds`);
        };
        
        const originalHandleCombinedSearch = handleCombinedSearch;
        handleCombinedSearch = async function() {
            searchStartTime = performance.now();
            await originalHandleCombinedSearch();
            const searchTime = ((performance.now() - searchStartTime) / 1000).toFixed(2);
            console.log(`Combined search completed in ${searchTime} seconds`);
        };
    </script>
</body>
</html>
    """

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Get users from Redis or fallback
        users = get_users()
        
        if username in users and users[username] == password:
            session['username'] = username
            session.permanent = True
            app.permanent_session_lifetime = 1800  # 30 minutes
            return redirect(url_for('index'))
        else:
            return render_template_string(get_login_template(), error="Invalid username or password. Please try again.")
    
    # If user is already logged in, redirect to main page
    if 'username' in session:
        return redirect(url_for('index'))
    
    return render_template_string(get_login_template())

def add_user_to_redis(username, password):
    """Add a user to Redis"""
    if redis_client:
        try:
            redis_client.hset("users", username, password)
            print(f"✅ User {username} added to Redis")
            return True
        except Exception as e:
            print(f"❌ Error adding user: {e}")
    return False

# Optional: Add a simple API endpoint to add users
@app.route('/api/add_user', methods=['POST'])
def api_add_user():
    """Simple endpoint to add users (no validation for simplicity)"""

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    if username and password:
        if add_user_to_redis(username, password):
            print("good")
            return jsonify({'message': f'User {username} added successfully'})
    
    return jsonify({'error': 'Failed to add user'}), 400

@app.route('/logout')
def logout():
    """Logout and clear session"""
    username = session.get('username', 'User')
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    """Serve the main application (protected)"""
    username = session.get('username', 'User')
    return render_template_string(get_main_template(), username=username)

@app.route('/api/search', methods=['POST'])
@login_required
def api_search():
    """API endpoint for image search (protected)"""
    if not searcher:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    try:
        data = request.get_json()
        query_name = data.get('query_name')
        models = data.get('models', AVAILABLE_MODELS)
        metric = data.get('metric', 'euclidean')
        top_k = int(data.get('top_k', 20))
        
        if not query_name:
            return jsonify({'error': 'Query name is required'}), 400
        
        results = {}
        pr_curves = {}
        
        for model_name in models:
            if model_name in searcher.models:
                try:
                    neighbors = searcher.get_k_neighbors(model_name, query_name, top_k, metric)
                    
                    # Convert results to JSON-serializable format
                    serializable_neighbors = []
                    for name, score in neighbors:
                        serializable_neighbors.append([str(name), float(score)])
                    
                    results[model_name] = serializable_neighbors
                    
                    # Calculate precision-recall
                    pr_data = calculate_precision_recall(query_name, neighbors)
                    pr_curve = create_pr_curve(pr_data, model_name)
                    if pr_curve:
                        pr_curves[model_name] = pr_curve
                    
                except Exception as e:
                    print(f"Error processing {model_name}: {e}")
                    results[model_name] = {'error': str(e)}
            else:
                results[model_name] = {'error': f'Model {model_name} not available'}
        
        response_data = {
            'query_name': str(query_name),
            'results': results,
            'pr_curves': pr_curves,
            'metric': str(metric),
            'timestamp': datetime.now().isoformat(),
            'user': session.get('username')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Search API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/available_images')
@login_required
def api_available_images():
    """Get list of available images (protected)"""
    if not searcher or not AVAILABLE_MODELS:
        return jsonify({'images': []})
    
    try:
        # Get available images from one of the models
        model_name = AVAILABLE_MODELS[0]
        images = list(searcher.models[model_name].keys())
        sorted_images = sorted(images, key=lambda x: int(x) if x.isdigit() else x)
        return jsonify({'images': sorted_images})
    except Exception as e:
        print(f"Error getting available images: {e}")
        return jsonify({'images': []})

@app.route('/api/stats')
@login_required
def api_stats():
    """Get application statistics (protected)"""
    if not searcher:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    try:
        stats = {
            'total_images': len(searcher.image_dict),
            'available_models': AVAILABLE_MODELS,
            'model_features': {model: len(features) for model, features in searcher.models.items()},
            'available_metrics': AVAILABLE_METRICS,
            'uptime': 'System running',
            'current_user': session.get('username')
        }
        
        return jsonify(stats)
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<path:filename>')
@login_required
def serve_image(filename):
    """Serve images from the image folder (protected)"""
    try:
        return send_from_directory('image.orig', filename)
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        return "Image not found", 404


def combine_model_results(results_dict, combination_method='average', weights=None):
    """
    Combine results from multiple models
    
    Args:
        results_dict: Dictionary of {model_name: [(image_name, score), ...]}
        combination_method: 'average', 'weighted', 'rank_fusion'
        weights: Dictionary of {model_name: weight} for weighted combination
    
    Returns:
        Combined results list
    """
    if len(results_dict) == 1:
        return list(results_dict.values())[0]
    
    # Get all unique images
    all_images = set()
    for results in results_dict.values():
        for image_name, _ in results:
            all_images.add(image_name)
    
    combined_scores = {}
    
    if combination_method == 'average':
        # Simple average of normalized scores
        for image in all_images:
            scores = []
            for model_name, results in results_dict.items():
                # Find score for this image in this model
                for img_name, score in results:
                    if img_name == image:
                        scores.append(score)
                        break
                else:
                    # Image not found in this model results, assign worst score
                    worst_score = max([s for _, s in results]) if results else 1.0
                    scores.append(worst_score)
            
            combined_scores[image] = sum(scores) / len(scores)
    
    elif combination_method == 'weighted' and weights:
        # Weighted combination
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        for image in all_images:
            weighted_score = 0
            for model_name, results in results_dict.items():
                weight = normalized_weights.get(model_name, 0)
                for img_name, score in results:
                    if img_name == image:
                        weighted_score += score * weight
                        break
                else:
                    # Image not found, use worst score
                    worst_score = max([s for _, s in results]) if results else 1.0
                    weighted_score += worst_score * weight
            
            combined_scores[image] = weighted_score
    
    elif combination_method == 'rank_fusion':
        # Rank-based fusion (lower rank = better)
        for image in all_images:
            rank_sum = 0
            for model_name, results in results_dict.items():
                # Find rank of this image in this model
                for rank, (img_name, score) in enumerate(results):
                    if img_name == image:
                        rank_sum += rank + 1  # 1-based ranking
                        break
                else:
                    # Image not found, assign worst rank
                    rank_sum += len(results) + 1
            
            combined_scores[image] = rank_sum / len(results_dict)
    
    # Sort results (lower score = better for distances, higher for similarities)
    # For consistency, we'll assume lower scores are better
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1])
    
    return sorted_results

def calculate_advanced_metrics(query_name, results, total_relevant=20):
    """Calculate additional metrics for combined results"""
    try:
        query_class = int(query_name) // 100 if query_name.isdigit() else 0
    except:
        query_class = 0
    
    relevant_found = 0
    for i, (image_name, score) in enumerate(results):
        try:
            result_class = int(image_name) // 100 if image_name.isdigit() else -1
        except:
            result_class = -1
        
        if result_class == query_class:
            relevant_found += 1
    
    precision_at_k = relevant_found / len(results) if results else 0
    recall_at_k = relevant_found / total_relevant if total_relevant > 0 else 0
    
    return {
        'precision_at_k': float(precision_at_k),
        'recall_at_k': float(recall_at_k),
        'relevant_found': relevant_found,
        'total_results': len(results)
    }


@app.route('/api/search_combined', methods=['POST'])
@login_required
def api_search_combined():
    """API endpoint for combined model search (protected)"""
    if not searcher:
        return jsonify({'error': 'Search engine not initialized'}), 500
    
    try:
        data = request.get_json()
        query_name = data.get('query_name')
        models = data.get('models', AVAILABLE_MODELS)
        metric = data.get('metric', 'euclidean')
        top_k = int(data.get('top_k', 20))
        combination_method = data.get('combination_method', 'average')
        model_weights = data.get('model_weights', {})
        
        print(query_name)
        print(models)
        print(metric)
        print(top_k)
        print(combination_method)
        if not query_name:
            return jsonify({'error': 'Query name is required'}), 400
        
        if len(models) < 2:
            return jsonify({'error': 'At least 2 models required for combination'}), 400
        
        # Get individual model results
        individual_results = {}
        for model_name in models:
            if model_name in searcher.models:
                try:
                    neighbors = searcher.get_k_neighbors(model_name, query_name, top_k, metric)
                    individual_results[model_name] = neighbors
                except Exception as e:
                    print(f"Error processing {model_name}: {e}")
                    continue
        
        if len(individual_results) < 2:
            return jsonify({'error': 'Not enough models available for combination'}), 400
        
        # Combine results
        combined_results = combine_model_results(
            individual_results, 
            combination_method, 
            model_weights if model_weights else None
        )
        
        # Calculate metrics for combined results
        pr_data = calculate_precision_recall(query_name, combined_results[:top_k])
        advanced_metrics = calculate_advanced_metrics(query_name, combined_results[:top_k])
        pr_curve = create_pr_curve(pr_data, f'Combined ({combination_method})')
        
        response_data = {
            'query_name': str(query_name),
            'individual_results': {k: [(str(name), float(score)) for name, score in v] 
                                 for k, v in individual_results.items()},
            'combined_results': [(str(name), float(score)) for name, score in combined_results[:top_k]],
            'combination_method': combination_method,
            'model_weights': model_weights,
            'pr_curve': pr_curve,
            'advanced_metrics': advanced_metrics,
            'metric': str(metric),
            'timestamp': datetime.now().isoformat(),
            'user': session.get('username')
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Combined search API error: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/health')
def health_check():
    """Simple health check for Docker and monitoring"""
    try:
        status_data = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'models_available': len(AVAILABLE_MODELS) if AVAILABLE_MODELS else 0,
            'images_loaded': len(searcher.image_dict) if searcher and searcher.image_dict else 0,
            'redis_connected': redis_client is not None
        }
        
        # Test Redis connection if available
        if redis_client:
            try:
                redis_client.ping()
                status_data['redis_status'] = 'connected'
            except:
                status_data['redis_status'] = 'disconnected'
        else:
            status_data['redis_status'] = 'not_configured'
            
        return jsonify(status_data)
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500





if __name__ == '__main__':
    print("🚀 Starting Multimedia Search Engine with Redis Authentication...")
    
    # Test Redis and show users
    users = get_users()
    print(f"👥 Available users: {list(users.keys())}")
    
    if redis_client:
        print(f"✅ Using Redis for authentication at {REDIS_HOST}:{REDIS_PORT}")
    else:
        print("⚠️  Using fallback authentication (hardcoded)")
    
    print(f"📊 Available models: {AVAILABLE_MODELS}")
    print(f"📊 Model count: {len(AVAILABLE_MODELS)}")
    if searcher:
        print(f"🖼️  Total images: {len(searcher.image_dict)}")
    print("🔐 Authentication enabled")
    print("🌐 Access the application at: http://localhost:5000")
    print("🔑 Login required to access the search engine")
    
    app.run(host='0.0.0.0', port=5000, debug=True)


