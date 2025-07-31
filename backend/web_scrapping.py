from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import threading

# Global lock for thread-safe operations
results_lock = Lock()

def setup_driver():
    chrome_options = Options() # Enable headless mode
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )

    driver = webdriver.Chrome(options=chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def read_multi_input(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Input file {file_path} not found!")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {file_path}")
        return []

def build_amazon_url(params):
    """Build Amazon search URL using all available product details"""
    # Start with the main product name
    search_terms = []
    
    # Add product name/hint
    product_name = params.get("product_hint") or params.get("garment") or ""
    if product_name:
        search_terms.append(product_name)
    
    # Add gender for more targeted results
    gender = params.get("gender", "")
    if gender and gender.lower() in ['male', 'female', 'men', 'women']:
        if gender.lower() in ['male', 'men']:
            search_terms.append("men")
        elif gender.lower() in ['female', 'women']:
            search_terms.append("women")
    
    # Add fit information
    fit = params.get("fit", "")
    if fit:
        # Extract size if present (e.g., "Relaxed Fit - M" -> "M")
        if " - " in fit:
            fit_parts = fit.split(" - ")
            fit_type = fit_parts[0].strip()
            size = fit_parts[1].strip() if len(fit_parts) > 1 else ""
            if fit_type:
                search_terms.append(fit_type.lower())
            if size and len(size) <= 3:  # Add size if it's reasonable (S, M, L, XL, etc.)
                search_terms.append(size)
        else:
            search_terms.append(fit.lower())
    
    # Add primary color from color palette
    color = params.get("color", "")
    if color and color != "N/A":
        # Take the first color if multiple colors are mentioned
        primary_color = color.split(',')[0].strip().split()[0]  # Get first word of first color
        if primary_color.lower() not in ['n/a', 'na', 'none']:
            search_terms.append(primary_color.lower())
    
    # Join all terms and format for URL
    search_query = " ".join(search_terms)
    search_query = re.sub(r'\s+', '+', search_query.strip())
    
    print(f"Built search URL for: {search_terms} -> {search_query}")
    return f"https://www.amazon.in/s?k={search_query}"

def safe_get_text(element, xpath, default="N/A"):
    try:
        for elem in element.find_elements(By.XPATH, xpath):
            text = elem.text.strip()
            if text:
                return text
        return default
    except:
        return default

def safe_get_attribute(element, xpath, attribute, default="N/A"):
    try:
        for elem in element.find_elements(By.XPATH, xpath):
            attr_value = elem.get_attribute(attribute)
            if attr_value and attr_value.strip():
                return attr_value.strip()
        return default
    except:
        return default

def extract_rating(product_container):
    try:
        star_elements = product_container.find_elements(By.XPATH, ".//i[contains(@class, 'a-icon-star')]")
        for star_elem in star_elements:
            try:
                parent = star_elem.find_element(By.XPATH, "./parent::*")
                aria_label = parent.get_attribute('aria-label')
                if aria_label and 'out of' in aria_label.lower():
                    return aria_label
            except:
                continue
    except:
        pass

    try:
        rating_elements = product_container.find_elements(By.XPATH, ".//span[@class='a-icon-alt']")
        for elem in rating_elements:
            text = elem.text.strip()
            if 'out of' in text.lower():
                return text
    except:
        pass

    try:
        star_elements = product_container.find_elements(By.XPATH, ".//i[contains(@class, 'a-icon-star')]")
        for star_elem in star_elements:
            class_name = star_elem.get_attribute('class') or ""
            match = re.search(r'a-icon-star-(\d+(?:-\d+)?)', class_name)
            if match:
                return f"{match.group(1).replace('-', '.')} out of 5 stars"
    except:
        pass

    return "N/A"

def scrape_product_data(driver, product_container, index):
    try:
        print(f"--- Scraping product {index} ---")

        rating = extract_rating(product_container)
        num_reviews = safe_get_text(product_container, ".//span[@class='a-size-base s-underline-text']", "N/A")
        price = safe_get_text(product_container, ".//span[@class='a-price-whole']", "N/A")
        if price.isdigit():
            price = f"₹{price}"

        brand = safe_get_text(product_container, ".//h2//span", "N/A")
        
        # Product description with multiple XPath patterns
        desc_xpaths = [
            ".//h2[contains(@class, 'a-size-mini')]//span[@class='a-size-base-plus']",
            ".//a[contains(@class, 'a-link-normal')]//h2//span",
            ".//span[@class='a-size-base-plus']"
        ]
        
        description = "N/A"
        for xpath in desc_xpaths:
            desc_text = safe_get_text(product_container, xpath, "N/A")
            if desc_text != "N/A" and desc_text.strip():
                description = desc_text
                break
        image_link = safe_get_attribute(product_container, ".//img[contains(@class, 's-image')]", "src", "N/A")
        product_link = safe_get_attribute(product_container, ".//a[contains(@class, 'a-link-normal')]", "href", "N/A")
        if product_link != "N/A" and not product_link.startswith("http"):
            product_link = f"https://www.amazon.in{product_link}"

        return {
            "rating": rating,
            "number_of_reviews": num_reviews,
            "price": price,
            "brand": brand,
            "description": description,
            "image_link": image_link,
            "product_link": product_link
        }

    except Exception as e:
        print(f"[ERROR] Error scraping product {index}: {e}")
        return None

def scrape_for_each_garment(driver, garment_data, max_products=7):
    url = build_amazon_url(garment_data)
    product_hint = garment_data.get('product_hint', 'Unknown product')
    print(f"\n[INFO] Searching for: {product_hint}")
    print(f"[URL] {url}")

    try:
        driver.get(url)
        time.sleep(3)
        
        # Wait for search results with increased timeout
        wait = WebDriverWait(driver, 15)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-component-type='s-search-result']")))

        product_containers = driver.find_elements(By.CSS_SELECTOR, "[data-component-type='s-search-result']")
        print(f"[FOUND] {len(product_containers)} products for '{product_hint}'")
        
        if not product_containers:
            print(f"[WARNING] No products found for '{product_hint}' - trying alternative search")
            return []

        results = []
        successful_products = 0
        
        for i, container in enumerate(product_containers[:max_products]):
            try:
                product_data = scrape_product_data(driver, container, i + 1)
                if product_data:
                    product_data['search_parameters'] = garment_data
                    product_data['product_index'] = i + 1
                    results.append(product_data)
                    successful_products += 1
                    print(f"[SCRAPED] Product {i + 1}: {product_data.get('description', 'Unknown')[:50]}...")
                else:
                    print(f"[SKIP] Product {i + 1}: Failed to extract data")
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"[ERROR] Failed to scrape product {i + 1}: {e}")
                continue

        print(f"[RESULT] Successfully scraped {successful_products}/{min(len(product_containers), max_products)} products for '{product_hint}'")
        return results
        
    except TimeoutException:
        print(f"[TIMEOUT] No search results found for '{product_hint}' within timeout period")
        return []
    except Exception as e:
        print(f"[ERROR] Failed to scrape '{product_hint}': {e}")
        return []

def scrape_single_garment_parallel(garment_data, max_products=7, thread_id=None):
    """
    Scrape a single garment in a separate thread with its own driver instance.
    This function is designed to be thread-safe and run in parallel.
    """
    thread_name = thread_id or threading.current_thread().name
    driver = None
    
    try:
        print(f"[THREAD-{thread_name}] Starting scraping for: {garment_data.get('product_hint', 'Unknown product')}")
        driver = setup_driver()
        
        garment_results = scrape_for_each_garment(driver, garment_data, max_products=max_products)
        
        if garment_results:
            print(f"[THREAD-{thread_name}] SUCCESS: Found {len(garment_results)} products")
            return {
                'success': True,
                'results': garment_results,
                'garment': garment_data.get('product_hint', 'Unknown'),
                'thread_id': thread_name
            }
        else:
            print(f"[THREAD-{thread_name}] WARNING: No products found")
            return {
                'success': False,
                'results': [],
                'garment': garment_data.get('product_hint', 'Unknown'),
                'thread_id': thread_name,
                'error': 'No products found'
            }
            
    except Exception as e:
        print(f"[THREAD-{thread_name}] ERROR: {str(e)}")
        return {
            'success': False,
            'results': [],
            'garment': garment_data.get('product_hint', 'Unknown'),
            'thread_id': thread_name,
            'error': str(e)
        }
    finally:
        if driver:
            try:
                driver.quit()
                print(f"[THREAD-{thread_name}] Browser closed")
            except Exception as e:
                print(f"[THREAD-{thread_name}] Error closing browser: {e}")

def rank_products_by_quality(products):
    """
    Rank products by quality based on rating, reviews, and other factors.
    Returns products sorted by quality score (highest first).
    """
    def calculate_quality_score(product):
        score = 0
        
        # Rating score (0-50 points)
        rating_text = product.get('rating', '0')
        try:
            # Extract numeric rating from text like "4.5 out of 5 stars"
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                rating = float(rating_match.group(1))
                score += rating * 10  # 5-star rating = 50 points
        except:
            pass
        
        # Review count score (0-30 points)
        reviews_text = product.get('number_of_reviews', '0')
        try:
            # Extract number from text like "1,234" or "1234"
            reviews_clean = re.sub(r'[^\d]', '', reviews_text)
            if reviews_clean:
                review_count = int(reviews_clean)
                # Logarithmic scale for review count (more reviews = better)
                if review_count > 0:
                    import math
                    score += min(30, math.log10(review_count) * 10)
        except:
            pass
        
        # Brand recognition bonus (0-10 points)
        brand = product.get('brand', '').lower()
        popular_brands = ['nike', 'adidas', 'puma', 'reebok', 'h&m', 'zara', 'uniqlo', 'levis', 'tommy']
        if any(popular_brand in brand for popular_brand in popular_brands):
            score += 10
        
        # Price reasonableness (0-10 points)
        price_text = product.get('price', '₹0')
        try:
            price_match = re.search(r'₹([\d,]+)', price_text)
            if price_match:
                price = int(price_match.group(1).replace(',', ''))
                # Reasonable price range gets bonus (₹500-₹3000)
                if 500 <= price <= 3000:
                    score += 10
                elif 300 <= price <= 5000:
                    score += 5
        except:
            pass
        
        return score
    
    # Calculate scores and sort
    for product in products:
        product['quality_score'] = calculate_quality_score(product)
    
    # Sort by quality score (highest first)
    ranked_products = sorted(products, key=lambda x: x.get('quality_score', 0), reverse=True)
    
    return ranked_products

def scrape_multi_garment_distributed_8(garments, output_file, max_workers=3):
    """
    Scrape multiple garments in parallel and return exactly 8 results distributed as:
    - Top 3 products from garment 1
    - Top 3 products from garment 2  
    - Top 2 products from garment 3
    This minimizes scraping time while ensuring quality distribution.
    """
    # If garments is a filename, load it; if it's a list, use as is
    if isinstance(garments, str):
        garments = read_multi_input(garments)
    if not garments:
        print("[WARNING] No garments to scrape")
        return

    # Ensure we have exactly 3 garments
    if len(garments) < 3:
        print(f"[WARNING] Expected 3 garments, got {len(garments)}. Padding with duplicates.")
        while len(garments) < 3:
            garments.append(garments[-1])  # Duplicate last garment
    elif len(garments) > 3:
        print(f"[INFO] Got {len(garments)} garments, using first 3 only.")
        garments = garments[:3]

    print(f"\n[PARALLEL-3+3+2] Starting efficient parallel scraping for 3 garments...")
    print(f"[CONFIG] Distribution: 3+3+2 = 8 total results, Max workers: {max_workers}")
    
    # Define how many products to scrape per garment (minimal for speed)
    products_needed = [3, 3, 2]  # Garment 1: 3, Garment 2: 3, Garment 3: 2
    products_to_scrape = [4, 4, 3]  # Scrape slightly more to have selection
    
    garment_results = [[] for _ in range(3)]  # Store results per garment
    successful_scrapes = 0
    failed_scrapes = 0
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit scraping tasks for each garment with specific product counts
        future_to_garment = {
            executor.submit(scrape_single_garment_parallel, garments[i], products_to_scrape[i], f"G{i+1}"): i
            for i in range(3)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_garment):
            garment_index = future_to_garment[future]
            
            try:
                result = future.result(timeout=90)  # 1.5 minute timeout per garment
                
                if result['success'] and result['results']:
                    # Rank products for this garment and take only what we need
                    ranked_products = rank_products_by_quality(result['results'])
                    needed_count = products_needed[garment_index]
                    selected_products = ranked_products[:needed_count]
                    
                    garment_results[garment_index] = selected_products
                    successful_scrapes += 1
                    
                    garment_name = garments[garment_index].get('product_hint', 'Unknown')
                    print(f"[COMPLETED] Garment {garment_index+1}: {garment_name} - Selected {len(selected_products)}/{needed_count} products")
                    
                    # Show selected products
                    for j, product in enumerate(selected_products, 1):
                        score = product.get('quality_score', 0)
                        brand = product.get('brand', 'Unknown')
                        rating = product.get('rating', 'N/A')
                        print(f"    {j}. {brand} - {rating} (Score: {score:.1f})")
                else:
                    failed_scrapes += 1
                    garment_name = garments[garment_index].get('product_hint', 'Unknown')
                    error_msg = result.get('error', 'No products found')
                    print(f"[FAILED] Garment {garment_index+1}: {garment_name} - {error_msg}")
                    
            except Exception as e:
                failed_scrapes += 1
                garment_name = garments[garment_index].get('product_hint', 'Unknown')
                print(f"[ERROR] Garment {garment_index+1}: {garment_name} - {str(e)}")
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Combine all results in order: G1(3) + G2(3) + G3(2)
    final_results = []
    for i, results in enumerate(garment_results):
        final_results.extend(results)
        if results:
            print(f"[ADDED] {len(results)} products from garment {i+1}")
    
    # Add metadata to track distribution
    distribution_info = {
        "total_products": len(final_results),
        "distribution": f"{len(garment_results[0])}+{len(garment_results[1])}+{len(garment_results[2])}",
        "garment_1_count": len(garment_results[0]),
        "garment_2_count": len(garment_results[1]),
        "garment_3_count": len(garment_results[2]),
        "scraping_time_seconds": round(total_time, 2)
    }
    
    # Save results in frontend-compatible format
    # Frontend expects 'scraped_products' key, so we'll provide both formats
    output_data = {
        "distribution_info": distribution_info,
        "products": final_results,
        "scraped_products": final_results  # Frontend compatibility
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {len(final_results)} distributed results saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return
    
    # Print summary
    print(f"\n[DISTRIBUTED SUMMARY] Scraping completed in {total_time:.2f} seconds:")
    print(f"[SUCCESS] {successful_scrapes}/3 garments scraped successfully")
    print(f"[FAILED] {failed_scrapes}/3 garments failed")
    print(f"[DISTRIBUTION] {len(garment_results[0])}+{len(garment_results[1])}+{len(garment_results[2])} = {len(final_results)} total products")
    print(f"[PERFORMANCE] Average time per garment: {total_time/3:.2f} seconds")
    print(f"[EFFICIENCY] Only scraped ~{sum(products_to_scrape)} products total vs previous ~21 products")
    
    return final_results

def scrape_multi_garment_distributed_10(garments, output_file, max_workers=3):
    """
    Scrape multiple garments in parallel and return exactly 10 results distributed as:
    - Top 4 products from garment 1
    - Top 3 products from garment 2
    - Top 3 products from garment 3
    This minimizes scraping time while ensuring quality distribution.
    """
    # If garments is a filename, load it; if it's a list, use as is
    if isinstance(garments, str):
        garments = read_multi_input(garments)
    if not garments:
        print("[WARNING] No garments to scrape")
        return

    # Ensure we have exactly 3 garments
    if len(garments) < 3:
        print(f"[WARNING] Expected 3 garments, got {len(garments)}. Padding with duplicates.")
        while len(garments) < 3:
            garments.append(garments[-1])  # Duplicate last garment
    elif len(garments) > 3:
        print(f"[INFO] Got {len(garments)} garments, using first 3 only.")
        garments = garments[:3]

    print(f"\n[PARALLEL-4+3+3] Starting efficient parallel scraping for 3 garments...")
    print(f"[CONFIG] Distribution: 4+3+3 = 10 total results, Max workers: {max_workers}")
    
    # Define how many products to scrape per garment (scrape more for better selection)
    products_needed = [4, 3, 3]  # Garment 1: 4, Garment 2: 3, Garment 3: 3
    products_to_scrape = [7, 7, 7]  # Scrape 7 per garment
    
    garment_results = [[] for _ in range(3)]  # Store results per garment
    successful_scrapes = 0
    failed_scrapes = 0
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit scraping tasks for each garment with specific product counts
        future_to_garment = {
            executor.submit(scrape_single_garment_parallel, garments[i], products_to_scrape[i], f"G{i+1}"): i
            for i in range(3)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_garment):
            garment_index = future_to_garment[future]
            try:
                result = future.result(timeout=120)  # 2 minute timeout per garment
                if result['success'] and result['results']:
                    # Rank products for this garment and take only what we need
                    ranked_products = rank_products_by_quality(result['results'])
                    needed_count = products_needed[garment_index]
                    selected_products = ranked_products[:needed_count]
                    garment_results[garment_index] = selected_products
                    successful_scrapes += 1
                    garment_name = garments[garment_index].get('product_hint', 'Unknown')
                    print(f"[COMPLETED] Garment {garment_index+1}: {garment_name} - Selected {len(selected_products)}/{needed_count} products")
                    for j, product in enumerate(selected_products, 1):
                        score = product.get('quality_score', 0)
                        brand = product.get('brand', 'Unknown')
                        rating = product.get('rating', 'N/A')
                        print(f"    {j}. {brand} - {rating} (Score: {score:.1f})")
                else:
                    failed_scrapes += 1
                    garment_name = garments[garment_index].get('product_hint', 'Unknown')
                    error_msg = result.get('error', 'No products found')
                    print(f"[FAILED] Garment {garment_index+1}: {garment_name} - {error_msg}")
            except Exception as e:
                failed_scrapes += 1
                garment_name = garments[garment_index].get('product_hint', 'Unknown')
                print(f"[ERROR] Garment {garment_index+1}: {garment_name} - {str(e)}")
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    # Combine all results in order: G1(4) + G2(3) + G3(3)
    final_results = []
    for i, results in enumerate(garment_results):
        final_results.extend(results)
        if results:
            print(f"[ADDED] {len(results)} products from garment {i+1}")
    # Add metadata to track distribution
    distribution_info = {
        "total_products": len(final_results),
        "distribution": f"{len(garment_results[0])}+{len(garment_results[1])}+{len(garment_results[2])}",
        "garment_1_count": len(garment_results[0]),
        "garment_2_count": len(garment_results[1]),
        "garment_3_count": len(garment_results[2]),
        "scraping_time_seconds": round(total_time, 2)
    }
    # Save results in frontend-compatible format
    output_data = {
        "distribution_info": distribution_info,
        "products": final_results,
        "scraped_products": final_results  # Frontend compatibility
    }
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] {len(final_results)} distributed results saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return
    # Print summary
    print(f"\n[DISTRIBUTED SUMMARY] Scraping completed in {total_time:.2f} seconds:")
    print(f"[SUCCESS] {successful_scrapes}/3 garments scraped successfully")
    print(f"[FAILED] {failed_scrapes}/3 garments failed")
    print(f"[DISTRIBUTION] {len(garment_results[0])}+{len(garment_results[1])}+{len(garment_results[2])} = {len(final_results)} total products")
    print(f"[PERFORMANCE] Average time per garment: {total_time/3:.2f} seconds")
    print(f"[EFFICIENCY] Only scraped ~{sum(products_to_scrape)} products total vs previous ~21 products")
    return final_results

def scrape_multi_garment_parallel_top7(garments, output_file, max_workers=3, target_results=7):
    """
    Legacy function - kept for compatibility.
    For efficient 3+3+2 distribution, use scrape_multi_garment_distributed_8() instead.
    """
    print("[INFO] Using legacy top-7 function. For better efficiency, use distributed 3+3+2 function.")
    
    # If garments is a filename, load it; if it's a list, use as is
    if isinstance(garments, str):
        garments = read_multi_input(garments)
    if not garments:
        print("[WARNING] No garments to scrape")
        return

    print(f"\n[PARALLEL-TOP7] Starting parallel scraping for {len(garments)} recommendations...")
    print(f"[CONFIG] Max workers: {max_workers}, Target top results: {target_results}")
    
    all_results = []
    successful_scrapes = 0
    failed_scrapes = 0
    start_time = time.time()
    
    # Scrape more products per garment to have better selection for top 7
    products_per_garment = max(5, target_results // len(garments) + 2)  # At least 5 per garment
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks
        future_to_garment = {
            executor.submit(scrape_single_garment_parallel, garment_data, products_per_garment, f"T{i+1}"): (i+1, garment_data)
            for i, garment_data in enumerate(garments)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_garment):
            garment_index, garment_data = future_to_garment[future]
            
            try:
                result = future.result(timeout=120)  # 2 minute timeout per garment
                
                if result['success']:
                    with results_lock:  # Thread-safe result collection
                        all_results.extend(result['results'])
                        successful_scrapes += 1
                    print(f"[COMPLETED] Garment {garment_index}/{len(garments)}: {result['garment']} - {len(result['results'])} products")
                else:
                    failed_scrapes += 1
                    print(f"[FAILED] Garment {garment_index}/{len(garments)}: {result['garment']} - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_scrapes += 1
                print(f"[ERROR] Garment {garment_index}/{len(garments)}: {garment_data.get('product_hint', 'Unknown')} - {str(e)}")
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Rank products and get top results
    if all_results:
        print(f"\n[RANKING] Ranking {len(all_results)} products by quality...")
        ranked_products = rank_products_by_quality(all_results)
        top_results = ranked_products[:target_results]
        
        print(f"[SELECTED] Top {len(top_results)} products selected:")
        for i, product in enumerate(top_results, 1):
            score = product.get('quality_score', 0)
            brand = product.get('brand', 'Unknown')
            rating = product.get('rating', 'N/A')
            price = product.get('price', 'N/A')
            print(f"  {i}. {brand} - {rating} - {price} (Score: {score:.1f})")
    else:
        print("[WARNING] No products found to rank")
        top_results = []
    
    # Save top results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(top_results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Top {len(top_results)} results saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return
    
    # Print summary
    print(f"\n[PARALLEL-TOP7 SUMMARY] Scraping completed in {total_time:.2f} seconds:")
    print(f"[SUCCESS] {successful_scrapes}/{len(garments)} recommendations scraped successfully")
    print(f"[FAILED] {failed_scrapes}/{len(garments)} recommendations failed")
    print(f"[COLLECTED] {len(all_results)} total products found across all recommendations")
    print(f"[FINAL] {len(top_results)} top-quality products selected and saved")
    print(f"[PERFORMANCE] Average time per garment: {total_time/len(garments):.2f} seconds")
    print(f"[SPEEDUP] Estimated sequential time would be ~{len(garments) * 15:.0f}s, parallel time: {total_time:.2f}s")
    
    return top_results

def scrape_multi_garment_parallel(garments, output_file, max_products=7, max_workers=3):
    """
    Legacy parallel scraping function - returns all products.
    For top-N results, use scrape_multi_garment_parallel_top7() instead.
    """
    # If garments is a filename, load it; if it's a list, use as is
    if isinstance(garments, str):
        garments = read_multi_input(garments)
    if not garments:
        print("[WARNING] No garments to scrape")
        return

    print(f"\n[PARALLEL] Starting parallel scraping for {len(garments)} recommendations...")
    print(f"[CONFIG] Max workers: {max_workers}, Max products per garment: {max_products}")
    
    all_results = []
    successful_scrapes = 0
    failed_scrapes = 0
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks
        future_to_garment = {
            executor.submit(scrape_single_garment_parallel, garment_data, max_products, f"T{i+1}"): (i+1, garment_data)
            for i, garment_data in enumerate(garments)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_garment):
            garment_index, garment_data = future_to_garment[future]
            
            try:
                result = future.result(timeout=120)  # 2 minute timeout per garment
                
                if result['success']:
                    with results_lock:  # Thread-safe result collection
                        all_results.extend(result['results'])
                        successful_scrapes += 1
                    print(f"[COMPLETED] Garment {garment_index}/{len(garments)}: {result['garment']} - {len(result['results'])} products")
                else:
                    failed_scrapes += 1
                    print(f"[FAILED] Garment {garment_index}/{len(garments)}: {result['garment']} - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_scrapes += 1
                print(f"[ERROR] Garment {garment_index}/{len(garments)}: {garment_data.get('product_hint', 'Unknown')} - {str(e)}")
    
    # Calculate timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVED] Results saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")
        return
    
    # Print summary
    print(f"\n[PARALLEL SUMMARY] Scraping completed in {total_time:.2f} seconds:")
    print(f"[SUCCESS] {successful_scrapes}/{len(garments)} recommendations scraped successfully")
    print(f"[FAILED] {failed_scrapes}/{len(garments)} recommendations failed")
    print(f"[TOTAL] {len(all_results)} products found across all recommendations")
    print(f"[PERFORMANCE] Average time per garment: {total_time/len(garments):.2f} seconds")
    print(f"[SPEEDUP] Estimated sequential time would be ~{len(garments) * 15:.0f}s, parallel time: {total_time:.2f}s")

def scrape_multi_garment(garments, output_file, max_products=7):
    """
    Legacy sequential scraping function - kept for compatibility.
    For better performance, use scrape_multi_garment_parallel() instead.
    """
    print("[INFO] Using legacy sequential scraping. For better performance, use parallel scraping.")
    
    # If garments is a filename, load it; if it's a list, use as is
    if isinstance(garments, str):
        garments = read_multi_input(garments)
    if not garments:
        print("[WARNING] No garments to scrape")
        return

    print(f"\n[INFO] Starting scraping for {len(garments)} recommendations...")
    all_results = []
    driver = setup_driver()
    successful_scrapes = 0
    failed_scrapes = 0

    try:
        for i, garment_data in enumerate(garments, 1):
            print(f"\n[PROGRESS] Processing recommendation {i}/{len(garments)}")
            print(f"[PRODUCT] {garment_data.get('product_hint', 'Unknown product')}")
            
            try:
                garment_results = scrape_for_each_garment(driver, garment_data, max_products=max_products)
                if garment_results:
                    all_results.extend(garment_results)
                    successful_scrapes += 1
                    print(f"[SUCCESS] Found {len(garment_results)} products for recommendation {i}")
                else:
                    failed_scrapes += 1
                    print(f"[WARNING] No products found for recommendation {i}")
                    
            except Exception as e:
                failed_scrapes += 1
                print(f"[ERROR] Failed to scrape recommendation {i}: {e}")
                continue

        # Save results even if some scraping failed
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n[SUMMARY] Scraping completed:")
        print(f"[SUCCESS] {successful_scrapes}/{len(garments)} recommendations scraped successfully")
        print(f"[TOTAL] {len(all_results)} products found across all recommendations")
        print(f"[OUTPUT] Results saved to: {output_file}")
        
        if failed_scrapes > 0:
            print(f"[WARNING] {failed_scrapes} recommendations failed to scrape")

    except TimeoutException:
        print("[ERROR] Timeout waiting for page elements")
        # Save partial results
        if all_results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[PARTIAL] Saved {len(all_results)} products from partial scraping")
    except Exception as e:
        print(f"[ERROR] Critical error during scraping: {e}")
        # Save partial results
        if all_results:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[PARTIAL] Saved {len(all_results)} products from partial scraping")
    finally:
        driver.quit()
        print("[CLOSED] Browser closed")

def llm_recs_to_garment_inputs(llm_recs, occasion="casual"):
    """
    Convert LLM recommendations to garment input format for scraping.
    Extracts all LLM fields: Product Name, Color Palette, Fit, Gender for URL construction
    """
    garment_inputs = []
    
    if not llm_recs:
        print("[WARNING] No LLM recommendations provided")
        return garment_inputs
    
    print(f"\n[INFO] Converting {len(llm_recs)} LLM recommendations to garment inputs...")
    
    for i, rec in enumerate(llm_recs, 1):
        # Extract color from color palette (preserve full palette for URL building)
        color_palette = rec.get('Color Palette', 'N/A')
        
        # Extract fit information (preserve full fit string including size)
        fit_info = rec.get('Fit', '')
        
        # Extract gender information
        gender_info = rec.get('Gender', '')
        
        # Extract product name
        product_name = rec.get('Product Name', '')
        
        # Validate that we have essential information
        if not product_name:
            print(f"[WARNING] Recommendation {i} missing product name, skipping")
            continue
            
        garment_input = {
            "product_hint": product_name,
            "color": color_palette,  # Pass full color palette to URL builder
            "fit": fit_info,         # Pass full fit info to URL builder
            "occasion": occasion,
            "gender": gender_info,   # Pass gender to URL builder
        }
        
        print(f"[CONVERT] Rec {i}: '{product_name}' -> {garment_input}")
        garment_inputs.append(garment_input)
    
    print(f"[SUCCESS] Converted {len(garment_inputs)} valid recommendations for scraping")
    return garment_inputs

def scrape_from_llm_recommendations(llm_json_file, output_file, max_products=7, occasion="casual"):
    print(f"\n[START] Starting web scraping from LLM recommendations...")
    print(f"[INPUT] LLM file: {llm_json_file}")
    print(f"[OUTPUT] Output file: {output_file}")
    print(f"[CONFIG] Max products per recommendation: {max_products}, Occasion: {occasion}")
    # Read LLM recommendations
    try:
        with open(llm_json_file, 'r', encoding='utf-8') as f:
            llm_recs = json.load(f)
        print(f"[LOADED] Successfully loaded {len(llm_recs) if isinstance(llm_recs, list) else 1} recommendations")
    except FileNotFoundError:
        print(f"[ERROR] LLM recommendations file not found: {llm_json_file}")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON format in {llm_json_file}: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Error reading {llm_json_file}: {e}")
        return
    # Validate LLM recommendations
    if not llm_recs:
        print("[ERROR] No LLM recommendations found in file")
        return
    if not isinstance(llm_recs, list):
        print("[ERROR] LLM recommendations should be a list")
        return
    # Convert to garment inputs with the provided occasion
    garment_inputs = llm_recs_to_garment_inputs(llm_recs, occasion)
    if not garment_inputs:
        print("[ERROR] No valid garments to scrape after conversion")
        return
    print(f"\n[READY] Starting efficient distributed parallel scraping for {len(garment_inputs)} valid recommendations...")
    # Use the new efficient 4+3+3 distributed function for minimal scraping time and 10 quality results
    scrape_multi_garment_distributed_10(garment_inputs, output_file, max_workers=3)

def create_sample_scraped_data():
    """Create sample scraped data for testing when real scraping fails"""
    sample_products = [
        {
            "brand": "Nike",
            "description": "Men's Cotton Casual T-Shirt - Navy Blue",
            "price": "₹899",
            "rating": "4.2 out of 5 stars",
            "number_of_reviews": "1,234",
            "image_link": "https://m.media-amazon.com/images/I/71HblAhdXxL.UX679.jpg",
            "product_link": "https://www.amazon.in/dp/B08N5WRWNW",
            "search_parameters": {"garment": "t-shirt", "color": "navy", "occasion": "casual"},
            "product_index": 1
        },
        {
            "brand": "Adidas",
            "description": "Men's Regular Fit Casual Shirt - White",
            "price": "₹1,299",
            "rating": "4.5 out of 5 stars", 
            "number_of_reviews": "856",
            "image_link": "https://m.media-amazon.com/images/I/61vFO3ijCeL.UX679.jpg",
            "product_link": "https://www.amazon.in/dp/B07QXZQZQZ",
            "search_parameters": {"garment": "shirt", "color": "white", "occasion": "casual"},
            "product_index": 2
        },
        {
            "brand": "Puma",
            "description": "Men's Slim Fit Jeans - Dark Blue Denim",
            "price": "₹1,599",
            "rating": "4.1 out of 5 stars",
            "number_of_reviews": "2,103",
            "image_link": "https://m.media-amazon.com/images/I/71YGQ5X8NFL.UX679.jpg",
            "product_link": "https://www.amazon.in/dp/B08XXXX123",
            "search_parameters": {"garment": "jeans", "color": "blue", "occasion": "casual"},
            "product_index": 3
        }
    ]
    return sample_products

# MAIN ENTRY
if __name__ == "__main__":
    input_file = "llm_recommendations.json"
    output_file = "multi_scraped_output.json"
    occasion = "casual"  # Default occasion, will be overridden by API
    
    print("[MAIN] Starting Fashion Recommendation Web Scraping (Parallel 4+3+3 Mode)")
    print(f"[MAIN] Input: {input_file}, Output: {output_file}, Occasion: {occasion}")
    
    scrape_from_llm_recommendations(input_file, output_file, max_products=7, occasion=occasion)