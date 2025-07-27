#!/usr/bin/env python3
"""
Comprehensive Backend Test Suite
Tests all components of the Fashion Recommendation System backend
"""

import requests
import json
import os
import sys
import time
from PIL import Image
import numpy as np
import traceback

class BackendTester:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.test_results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    def print_header(self, title):
        """Print a formatted test section header"""
        print("\n" + "="*60)
        print(f"🧪 {title}")
        print("="*60)
    
    def print_test(self, test_name, status, details=""):
        """Print individual test result"""
        self.total_tests += 1
        if status:
            self.passed_tests += 1
            print(f"✅ {test_name}")
            if details:
                print(f"   {details}")
        else:
            print(f"❌ {test_name}")
            if details:
                print(f"   {details}")
        
        self.test_results[test_name] = {"status": status, "details": details}
    
    def create_test_image(self):
        """Create a simple test image"""
        try:
            # Create a simple 200x200 RGB image
            img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            test_image_path = "test_image.jpg"
            img.save(test_image_path)
            return test_image_path
        except Exception as e:
            print(f"❌ Failed to create test image: {e}")
            return None
    
    def test_1_api_server_health(self):
        """Test if API server is running and responding"""
        self.print_header("API Server Health Check")
        
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.print_test("API Server Running", True, f"Status: {data.get('status', 'Unknown')}")
                self.print_test("Health Endpoint", True, f"Message: {data.get('message', 'No message')}")
                return True
            else:
                self.print_test("API Server Running", False, f"Status Code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            self.print_test("API Server Running", False, "Connection refused - server not running")
            return False
        except Exception as e:
            self.print_test("API Server Running", False, f"Error: {e}")
            return False
    
    def test_2_dependencies(self):
        """Test if all required dependencies are available"""
        self.print_header("Dependency Check")
        
        dependencies = {
            "torch": "PyTorch",
            "torchvision": "TorchVision", 
            "cv2": "OpenCV",
            "PIL": "Pillow",
            "timm": "TIMM",
            "joblib": "Joblib",
            "numpy": "NumPy",
            "flask": "Flask",
            "flask_cors": "Flask-CORS",
            "selenium": "Selenium",
            "webdriver_manager": "WebDriver Manager"
        }
        
        for module, name in dependencies.items():
            try:
                __import__(module)
                self.print_test(f"{name} Import", True)
            except ImportError as e:
                self.print_test(f"{name} Import", False, f"Import Error: {e}")
    
    def test_3_model_files(self):
        """Test if required model files exist"""
        self.print_header("Model Files Check")
        
        required_files = {
            "age_gender_model.pth": "Age/Gender Model",
            "age_scaler.pkl": "Age Scaler",
            "celeba_imbalance_aware_classifier.pth": "Facial Features Model"
        }
        
        for filename, description in required_files.items():
            if os.path.exists(filename):
                size = os.path.getsize(filename) / (1024*1024)  # Size in MB
                self.print_test(f"{description}", True, f"Size: {size:.1f} MB")
            else:
                self.print_test(f"{description}", False, "File not found")
    
    def test_4_environment_files(self):
        """Test if environment and configuration files exist"""
        self.print_header("Environment Files Check")
        
        files_to_check = {
            ".env": "Environment Variables",
            "list_attr_celeba.csv": "CelebA Attributes (Optional)",
        }
        
        for filename, description in files_to_check.items():
            if os.path.exists(filename):
                self.print_test(f"{description}", True, "File exists")
            else:
                self.print_test(f"{description}", False, "File not found (may be optional)")
    
    def test_5_individual_components(self):
        """Test individual backend components"""
        self.print_header("Individual Component Tests")
        
        # Test predict.py imports
        try:
            from predict import AgeGenderModel, process_image_for_recommendations
            self.print_test("Predict Module Import", True)
            
            # Test model initialization
            try:
                model = AgeGenderModel()
                self.print_test("AgeGenderModel Creation", True)
            except Exception as e:
                self.print_test("AgeGenderModel Creation", False, f"Error: {e}")
                
        except Exception as e:
            self.print_test("Predict Module Import", False, f"Error: {e}")
        
        # Test web scraping module
        try:
            from web_scrapping import setup_driver, create_sample_scraped_data
            self.print_test("Web Scraping Module Import", True)
            
            # Test sample data creation
            try:
                sample_data = create_sample_scraped_data()
                if len(sample_data) > 0:
                    self.print_test("Sample Data Creation", True, f"Generated {len(sample_data)} products")
                else:
                    self.print_test("Sample Data Creation", False, "No data generated")
            except Exception as e:
                self.print_test("Sample Data Creation", False, f"Error: {e}")
                
        except Exception as e:
            self.print_test("Web Scraping Module Import", False, f"Error: {e}")
        
        # Test LLM module
        try:
            from llm import run_fashion_llm
            self.print_test("LLM Module Import", True)
        except Exception as e:
            self.print_test("LLM Module Import", False, f"Error: {e}")
    
    def test_6_api_endpoints(self):
        """Test API endpoints with sample data"""
        self.print_header("API Endpoints Test")
        
        # Test upload-test endpoint
        test_image_path = self.create_test_image()
        if not test_image_path:
            self.print_test("Test Image Creation", False, "Could not create test image")
            return
        
        try:
            with open(test_image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(f"{self.base_url}/api/upload-test", files=files, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                self.print_test("Upload Test Endpoint", True, f"Success: {data.get('success', False)}")
            else:
                self.print_test("Upload Test Endpoint", False, f"Status: {response.status_code}")
                
        except Exception as e:
            self.print_test("Upload Test Endpoint", False, f"Error: {e}")
        
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    
    def test_7_full_pipeline(self):
        """Test the complete recommendation pipeline"""
        self.print_header("Full Pipeline Test")
        
        # Create test image
        test_image_path = self.create_test_image()
        if not test_image_path:
            self.print_test("Pipeline Test Setup", False, "Could not create test image")
            return
        
        try:
            # Prepare test data
            files = {'image': open(test_image_path, 'rb')}
            data = {
                'height': 170,
                'weight': 65,
                'occasion': 'casual'
            }
            
            print("🚀 Starting full pipeline test (this may take 30-60 seconds)...")
            
            # Send request
            response = requests.post(
                f"{self.base_url}/api/recommend", 
                files=files, 
                data=data, 
                timeout=120  # 2 minutes timeout
            )
            
            files['image'].close()
            
            if response.status_code == 200:
                result = response.json()
                
                # Check response structure
                success = result.get('success', False)
                self.print_test("Pipeline Execution", success, f"API returned success: {success}")
                
                # Check user info
                user_info = result.get('user_info', {})
                if user_info:
                    bmi = user_info.get('bmi', 0)
                    self.print_test("User Info Processing", True, f"BMI calculated: {bmi}")
                else:
                    self.print_test("User Info Processing", False, "No user info returned")
                
                # Check facial features
                facial_features = result.get('facial_features', {})
                if facial_features:
                    self.print_test("Facial Feature Analysis", True, f"Features detected: {len(facial_features)}")
                else:
                    self.print_test("Facial Feature Analysis", False, "No facial features returned")
                
                # Check LLM recommendations
                llm_recs = result.get('llm_recommendations', [])
                if llm_recs:
                    self.print_test("LLM Recommendations", True, f"Recommendations: {len(llm_recs)}")
                else:
                    self.print_test("LLM Recommendations", False, "No LLM recommendations returned")
                
                # Check scraped products
                products = result.get('scraped_products', [])
                if products:
                    self.print_test("Product Scraping", True, f"Products found: {len(products)}")
                    
                    # Check product structure
                    if len(products) > 0:
                        first_product = products[0]
                        required_fields = ['brand', 'description', 'price', 'rating']
                        missing_fields = [field for field in required_fields if not first_product.get(field) or first_product.get(field) == 'N/A']
                        
                        if not missing_fields:
                            self.print_test("Product Data Quality", True, "All required fields present")
                        else:
                            self.print_test("Product Data Quality", False, f"Missing fields: {missing_fields}")
                else:
                    self.print_test("Product Scraping", False, "No products returned")
                
                # Check total products
                total_products = result.get('total_products', 0)
                if total_products >= 5:
                    self.print_test("Product Count", True, f"Total products: {total_products}")
                else:
                    self.print_test("Product Count", False, f"Only {total_products} products (expected 5+)")
                    
            else:
                self.print_test("Pipeline Execution", False, f"HTTP {response.status_code}: {response.text[:200]}")
                
        except requests.exceptions.Timeout:
            self.print_test("Pipeline Execution", False, "Request timed out (>2 minutes)")
        except Exception as e:
            self.print_test("Pipeline Execution", False, f"Error: {e}")
            traceback.print_exc()
        
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    
    def test_8_output_files(self):
        """Check if output files are created properly"""
        self.print_header("Output Files Check")
        
        expected_files = {
            "multi_scraped_output.json": "Scraped Products",
            "llm_recommendations.json": "LLM Recommendations", 
            "facial_features.json": "Facial Features"
        }
        
        for filename, description in expected_files.items():
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        count = len(data.keys())
                    else:
                        count = 1
                    
                    self.print_test(f"{description} File", True, f"Contains {count} items")
                except Exception as e:
                    self.print_test(f"{description} File", False, f"Invalid JSON: {e}")
            else:
                self.print_test(f"{description} File", False, "File not found")
    
    def run_all_tests(self):
        """Run all tests and provide summary"""
        print("🧪 Fashion Recommendation Backend Test Suite")
        print("=" * 60)
        print("Testing all backend components...")
        
        start_time = time.time()
        
        # Run all tests
        server_running = self.test_1_api_server_health()
        
        if server_running:
            self.test_2_dependencies()
            self.test_3_model_files()
            self.test_4_environment_files()
            self.test_5_individual_components()
            self.test_6_api_endpoints()
            self.test_7_full_pipeline()
            self.test_8_output_files()
        else:
            print("\n⚠️  API Server not running - skipping remaining tests")
            print("Start the server with: python api.py")
        
        # Print summary
        end_time = time.time()
        duration = end_time - start_time
        
        self.print_header("TEST SUMMARY")
        print(f"📊 Tests Run: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.total_tests - self.passed_tests}")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("\n🎉 BACKEND IS READY FOR FRONTEND INTEGRATION!")
        elif success_rate >= 60:
            print("\n⚠️  BACKEND HAS SOME ISSUES - CHECK FAILED TESTS")
        else:
            print("\n❌ BACKEND NEEDS SIGNIFICANT FIXES BEFORE INTEGRATION")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if not server_running:
            print("   1. Start API server: python api.py")
        
        failed_tests = [name for name, result in self.test_results.items() if not result['status']]
        if failed_tests:
            print("   2. Fix failed tests:")
            for test in failed_tests[:5]:  # Show first 5 failed tests
                print(f"      - {test}")
        
        if success_rate >= 80:
            print("   3. Backend is ready - proceed with frontend integration!")
        
        return success_rate >= 80

def main():
    """Main function to run backend tests"""
    tester = BackendTester()
    
    print("🔍 Checking if backend server is running...")
    print("   Make sure you have started: python api.py")
    print("   Server should be running on: http://localhost:5000")
    
    input("\nPress Enter to start testing (or Ctrl+C to cancel)...")
    
    success = tester.run_all_tests()
    
    if success:
        print("\n🚀 Ready to proceed with frontend integration!")
    else:
        print("\n🔧 Please fix the issues above before frontend integration.")
    
    return success

if __name__ == "__main__":
    main()
