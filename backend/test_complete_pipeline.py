#!/usr/bin/env python3
"""
Test script to verify the complete fashion recommendation pipeline
This simulates what happens when frontend sends a request to backend
"""

import requests
import json
import os
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test image for testing"""
    # Create a simple 200x200 RGB image
    img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    test_image_path = "test_image.jpg"
    img.save(test_image_path)
    return test_image_path

def test_complete_pipeline():
    """Test the complete fashion recommendation pipeline"""
    
    print("🧪 Testing Complete Fashion Recommendation Pipeline")
    print("=" * 60)
    
    # Create test image
    print("📸 Creating test image...")
    test_image_path = create_test_image()
    
    # Test parameters
    test_params = {
        'height': 170,
        'weight': 65,
        'occasion': 'casual'
    }
    
    print(f"📋 Test Parameters:")
    print(f"   Height: {test_params['height']} cm")
    print(f"   Weight: {test_params['weight']} kg")
    print(f"   Occasion: {test_params['occasion']}")
    
    try:
        # Prepare request
        files = {'image': open(test_image_path, 'rb')}
        data = test_params
        
        print("\n🚀 Sending request to backend...")
        print("   URL: http://localhost:5000/api/recommend")
        
        # Send request to backend
        response = requests.post(
            'http://localhost:5000/api/recommend',
            files=files,
            data=data,
            timeout=300  # 5 minutes timeout for complete processing
        )
        
        files['image'].close()
        
        print(f"📡 Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS! Pipeline completed successfully")
            print("=" * 60)
            
            # Display results summary
            print(f"🎯 Success: {result.get('success', False)}")
            print(f"👤 User BMI: {result.get('user_info', {}).get('bmi', 'N/A')}")
            print(f"🛍️  Total Products Found: {result.get('total_products', 0)}")
            
            # Display facial features if available
            if result.get('facial_features'):
                print(f"\n👁️  Facial Features Detected:")
                for feature, value in result['facial_features'].items():
                    print(f"   {feature}: {value}")
            
            # Display LLM recommendations if available
            if result.get('llm_recommendations'):
                print(f"\n🤖 LLM Recommendations ({len(result['llm_recommendations'])}):")
                for i, rec in enumerate(result['llm_recommendations'][:3], 1):
                    print(f"   {i}. {rec.get('Garment Type', 'N/A')} - {rec.get('Color Palette', 'N/A')}")
            
            # Display scraped products
            if result.get('scraped_products'):
                print(f"\n🛒 Scraped Products ({len(result['scraped_products'])}):")
                for i, product in enumerate(result['scraped_products'][:5], 1):
                    print(f"   {i}. {product.get('brand', 'N/A')} - {product.get('price', 'N/A')}")
                    print(f"      Rating: {product.get('rating', 'N/A')}")
                    print(f"      Description: {product.get('description', 'N/A')[:50]}...")
                    if product.get('product_link'):
                        print(f"      Link: {product['product_link'][:60]}...")
                    print()
                
                # Save sample output for frontend reference
                with open('sample_frontend_response.json', 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print("💾 Sample response saved to: sample_frontend_response.json")
                
            else:
                print("⚠️  No scraped products found")
                
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - this is normal for first run as models load")
        print("   Try again, subsequent requests should be faster")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure backend server is running")
        print("   Run: python api.py")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"🧹 Cleaned up test image: {test_image_path}")

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running and healthy")
            return True
        else:
            print(f"⚠️  Backend responded with status: {response.status_code}")
            return False
    except:
        print("❌ Backend is not running or not accessible")
        print("   Start backend with: python api.py")
        return False

if __name__ == "__main__":
    print("🔍 Checking backend health...")
    if check_backend_health():
        print("\n" + "=" * 60)
        test_complete_pipeline()
    else:
        print("\n❌ Cannot proceed without backend running")
        print("Please start the backend server first: python api.py")
