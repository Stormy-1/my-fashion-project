import requests
import json

def test_api_health():
    """Test the health endpoint"""
    try:
        response = requests.get('http://localhost:5000/health')
        print("Health Check Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_fashion_recommendation(image_path, height, weight, occasion='casual'):
    """Test the fashion recommendation endpoint"""
    try:
        # Prepare the files and data
        files = {
            'image': open(image_path, 'rb')
        }
        data = {
            'height': height,
            'weight': weight,
            'occasion': occasion
        }
        
        print(f"\nTesting fashion recommendation with:")
        print(f"Image: {image_path}")
        print(f"Height: {height} cm")
        print(f"Weight: {weight} kg")
        print(f"Occasion: {occasion}")
        
        # Make the request
        response = requests.post('http://localhost:5000/recommend', files=files, data=data)
        
        print(f"\nAPI Response:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success: {result.get('success', False)}")
            print(f"Total Products Found: {result.get('total_products', 0)}")
            
            if 'user_info' in result:
                print(f"\nUser Info:")
                for key, value in result['user_info'].items():
                    print(f"  {key}: {value}")
            
            if 'facial_features' in result and result['facial_features']:
                print(f"\nFacial Features:")
                for key, value in result['facial_features'].items():
                    print(f"  {key}: {value}")
            
            if 'llm_recommendations' in result and result['llm_recommendations']:
                print(f"\nLLM Recommendations:")
                for i, rec in enumerate(result['llm_recommendations'][:2]):  # Show first 2
                    print(f"  Recommendation {i+1}:")
                    for key, value in rec.items():
                        print(f"    {key}: {value}")
            
            if 'scraped_products' in result and result['scraped_products']:
                print(f"\nScraped Products (first 3):")
                for i, product in enumerate(result['scraped_products'][:3]):
                    print(f"  Product {i+1}:")
                    print(f"    Brand: {product.get('brand', 'N/A')}")
                    print(f"    Description: {product.get('description', 'N/A')[:100]}...")
                    print(f"    Price: {product.get('price', 'N/A')}")
                    print(f"    Rating: {product.get('rating', 'N/A')}")
                    if product.get('product_link'):
                        print(f"    Link: {product['product_link'][:80]}...")
        else:
            print(f"Error Response: {response.text}")
        
        files['image'].close()
        return response.status_code == 200
        
    except Exception as e:
        print(f"Fashion recommendation test failed: {e}")
        return False

def test_upload_functionality(image_path):
    """Test the upload test endpoint"""
    try:
        files = {
            'image': open(image_path, 'rb')
        }
        
        response = requests.post('http://localhost:5000/upload-test', files=files)
        
        print(f"\nUpload Test Response:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        files['image'].close()
        return response.status_code == 200
        
    except Exception as e:
        print(f"Upload test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Fashion Recommendation API Test Client ===")
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint...")
    health_ok = test_api_health()
    
    if not health_ok:
        print("❌ Health check failed. Make sure the API server is running.")
        print("Run: python api.py")
        exit(1)
    
    print("✅ Health check passed!")
    
    # You'll need to provide a test image path
    test_image_path = input("\nEnter path to test image (or press Enter to skip): ").strip()
    
    if test_image_path and test_image_path.lower() != 'skip':
        # Test upload functionality
        print("\n2. Testing Upload Functionality...")
        upload_ok = test_upload_functionality(test_image_path)
        
        if upload_ok:
            print("✅ Upload test passed!")
        else:
            print("❌ Upload test failed!")
        
        # Test full recommendation pipeline
        print("\n3. Testing Fashion Recommendation Pipeline...")
        height = float(input("Enter height in cm (e.g., 170): ") or "170")
        weight = float(input("Enter weight in kg (e.g., 65): ") or "65")
        occasion = input("Enter occasion (e.g., casual, formal, party): ") or "casual"
        
        rec_ok = test_fashion_recommendation(test_image_path, height, weight, occasion)
        
        if rec_ok:
            print("✅ Fashion recommendation test completed!")
        else:
            print("❌ Fashion recommendation test failed!")
    else:
        print("Skipping image-based tests.")
    
    print("\n=== Test Summary ===")
    print("API is ready for frontend integration!")
    print("\nFrontend Integration Guide:")
    print("1. Upload image to POST /recommend")
    print("2. Include height, weight, occasion as form data")
    print("3. Receive JSON response with recommendations")
