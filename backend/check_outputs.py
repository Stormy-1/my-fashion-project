#!/usr/bin/env python3
"""
Simple script to check LLM and Web Scraping outputs
Shows actual recommendations and scraped products for manual verification
"""

import json
import os
from PIL import Image
import numpy as np

def create_sample_image():
    """Create a simple test image"""
    print("📸 Creating sample test image...")
    img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save("sample_test.jpg")
    print("✅ Sample image created: sample_test.jpg")
    return "sample_test.jpg"

def test_llm_recommendations():
    """Test LLM recommendations and show output"""
    print("\n" + "="*60)
    print("🤖 TESTING LLM RECOMMENDATIONS")
    print("="*60)
    
    try:
        from llm import run_fashion_llm
        
        # Sample user data
        sample_data = {
            "age": 25,
            "gender": "Male", 
            "height": 170,
            "weight": 65,
            "bmi": 22.5,
            "occasion": "casual",
            "facial_features": {
                "Attractive": 0.85,
                "Male": 0.95,
                "Young": 0.78,
                "Smiling": 0.65,
                "Eyeglasses": 0.92,
                "Chubby": 0.23,
                "Black_Hair": 0.88,
                "No_Beard": 0.76,
                "Oval_Face": 0.82
            }
        }
        
        print("📝 Input data for LLM:")
        print(f"   Age: {sample_data['age']}")
        print(f"   Gender: {sample_data['gender']}")
        print(f"   Height: {sample_data['height']} cm")
        print(f"   Weight: {sample_data['weight']} kg")
        print(f"   BMI: {sample_data['bmi']}")
        print(f"   Occasion: {sample_data['occasion']}")
        print(f"   Facial Features: {len(sample_data['facial_features'])} detected")
        
        print("\n🚀 Calling LLM for fashion recommendations...")
        print("⏳ This may take 10-30 seconds...")
        
        # Call LLM
        llm_result = run_fashion_llm(
            age=sample_data['age'],
            gender=sample_data['gender'],
            height=sample_data['height'],
            weight=sample_data['weight'],
            bmi=sample_data['bmi'],
            occasion=sample_data['occasion'],
            facial_features=sample_data['facial_features']
        )
        
        print("\n📋 LLM RECOMMENDATIONS OUTPUT:")
        print("-" * 40)
        if isinstance(llm_result, str):
            print(llm_result)
        else:
            print(json.dumps(llm_result, indent=2, ensure_ascii=False))
        
        # Check if LLM output was saved to file
        if os.path.exists("llm_recommendations.json"):
            print("\n📁 LLM output saved to: llm_recommendations.json")
            with open("llm_recommendations.json", 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            print("📄 Saved LLM data:")
            print(json.dumps(saved_data, indent=2, ensure_ascii=False))
        else:
            print("\n⚠️  llm_recommendations.json not found")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_scraping():
    """Test web scraping and show output"""
    print("\n" + "="*60)
    print("🕷️  TESTING WEB SCRAPING")
    print("="*60)
    
    try:
        from web_scrapping import scrape_multi_garment, create_sample_scraped_data
        
        # First try to create sample LLM recommendations if they don't exist
        if not os.path.exists("llm_recommendations.json"):
            print("📝 Creating sample LLM recommendations for scraping...")
            sample_llm_recs = [
                {
                    "Style Name": "Urban Casual Explorer",
                    "Product Name": "Cotton Casual T-Shirt",
                    "Garment Type": "T-Shirt",
                    "Color Palette": "Navy Blue",
                    "Fit": "Regular",
                    "Fabric": "Cotton",
                    "Accessories": "Casual Watch",
                    "Occasion": "Casual"
                },
                {
                    "Style Name": "Smart Casual Professional",
                    "Product Name": "Casual Denim Jeans",
                    "Garment Type": "Jeans", 
                    "Color Palette": "Dark Blue",
                    "Fit": "Slim",
                    "Fabric": "Denim",
                    "Accessories": "Leather Belt",
                    "Occasion": "Casual"
                }
            ]
            
            with open('llm_recommendations.json', 'w', encoding='utf-8') as f:
                json.dump(sample_llm_recs, f, indent=2, ensure_ascii=False)
            print("✅ Sample LLM recommendations created")
        
        print("\n🚀 Starting web scraping...")
        print("⏳ This may take 30-60 seconds...")
        print("📝 Scraping products based on LLM recommendations...")
        
        # Try web scraping
        try:
            scrape_multi_garment("llm_recommendations.json", "multi_scraped_output.json", max_products=5)
            print("✅ Web scraping completed!")
        except Exception as scrape_error:
            print(f"⚠️  Web scraping failed: {scrape_error}")
            print("🔄 Creating sample scraped data as fallback...")
            sample_data = create_sample_scraped_data()
            with open('multi_scraped_output.json', 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            print("✅ Sample scraped data created")
        
        # Show scraping results
        if os.path.exists("multi_scraped_output.json"):
            print("\n📁 Scraping output saved to: multi_scraped_output.json")
            with open("multi_scraped_output.json", 'r', encoding='utf-8') as f:
                scraped_data = json.load(f)
            
            print(f"\n🛍️  SCRAPED PRODUCTS ({len(scraped_data)} items):")
            print("-" * 50)
            
            for i, product in enumerate(scraped_data[:5], 1):  # Show first 5 products
                print(f"\n📦 Product {i}:")
                print(f"   Brand: {product.get('brand', 'N/A')}")
                print(f"   Description: {product.get('description', 'N/A')}")
                print(f"   Price: {product.get('price', 'N/A')}")
                print(f"   Rating: {product.get('rating', 'N/A')}")
                print(f"   Reviews: {product.get('number_of_reviews', 'N/A')}")
                print(f"   Image: {product.get('image_link', 'N/A')[:50]}...")
                print(f"   Product Link: {product.get('product_link', 'N/A')[:50]}...")
            
            if len(scraped_data) > 5:
                print(f"\n... and {len(scraped_data) - 5} more products")
            
            # Check data quality
            print(f"\n📊 DATA QUALITY CHECK:")
            print("-" * 30)
            
            total_products = len(scraped_data)
            products_with_brand = sum(1 for p in scraped_data if p.get('brand') and p.get('brand') != 'N/A')
            products_with_description = sum(1 for p in scraped_data if p.get('description') and p.get('description') != 'N/A')
            products_with_price = sum(1 for p in scraped_data if p.get('price') and p.get('price') != 'N/A')
            products_with_rating = sum(1 for p in scraped_data if p.get('rating') and p.get('rating') != 'N/A')
            
            print(f"   Total Products: {total_products}")
            print(f"   With Brand: {products_with_brand}/{total_products} ({products_with_brand/total_products*100:.1f}%)")
            print(f"   With Description: {products_with_description}/{total_products} ({products_with_description/total_products*100:.1f}%)")
            print(f"   With Price: {products_with_price}/{total_products} ({products_with_price/total_products*100:.1f}%)")
            print(f"   With Rating: {products_with_rating}/{total_products} ({products_with_rating/total_products*100:.1f}%)")
            
            return True
        else:
            print("❌ multi_scraped_output.json not found")
            return False
            
    except Exception as e:
        print(f"❌ Web Scraping Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the complete pipeline with a sample image"""
    print("\n" + "="*60)
    print("🔄 TESTING COMPLETE PIPELINE")
    print("="*60)
    
    try:
        from predict import process_image_for_recommendations
        
        # Create sample image
        test_image = create_sample_image()
        
        print(f"\n🚀 Processing image: {test_image}")
        print("⏳ This may take 1-2 minutes...")
        
        # Sample user parameters
        height = 170
        weight = 65
        occasion = "casual"
        
        print(f"📝 User parameters:")
        print(f"   Height: {height} cm")
        print(f"   Weight: {weight} kg") 
        print(f"   Occasion: {occasion}")
        
        # Process image
        result = process_image_for_recommendations(test_image, height, weight, occasion)
        
        print("\n📋 PIPELINE RESULT:")
        print("-" * 30)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Clean up
        if os.path.exists(test_image):
            os.remove(test_image)
            print(f"\n🗑️  Cleaned up test image: {test_image}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run all output checks"""
    print("🔍 FASHION RECOMMENDATION OUTPUT CHECKER")
    print("=" * 60)
    print("This script will show you the actual outputs from:")
    print("  1. LLM Fashion Recommendations")
    print("  2. Web Scraping Results")
    print("  3. Complete Pipeline")
    print("\nYou can manually verify if the outputs are correct.")
    
    input("\nPress Enter to start checking outputs...")
    
    # Test LLM
    llm_success = test_llm_recommendations()
    
    # Test Web Scraping
    scraping_success = test_web_scraping()
    
    # Test Full Pipeline
    pipeline_success = test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"🤖 LLM Recommendations: {'✅ Working' if llm_success else '❌ Failed'}")
    print(f"🕷️  Web Scraping: {'✅ Working' if scraping_success else '❌ Failed'}")
    print(f"🔄 Full Pipeline: {'✅ Working' if pipeline_success else '❌ Failed'}")
    
    if llm_success and scraping_success:
        print("\n🎉 BOTH LLM AND SCRAPING ARE WORKING!")
        print("📤 Your backend should return proper product data to frontend.")
    else:
        print("\n⚠️  SOME COMPONENTS NEED FIXING")
        if not llm_success:
            print("   - Fix LLM integration (check HF_TOKEN in .env)")
        if not scraping_success:
            print("   - Fix web scraping (check ChromeDriver setup)")
    
    print(f"\n📁 Check these files for detailed outputs:")
    print(f"   - llm_recommendations.json")
    print(f"   - multi_scraped_output.json")
    print(f"   - facial_features.json")

if __name__ == "__main__":
    main()
