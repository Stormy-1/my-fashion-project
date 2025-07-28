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

def setup_driver():
    chrome_options = Options() # Enable headless mode
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
    search_query = params.get("product_hint") or params.get("garment") or ""
    search_query = re.sub(r'\s+', '+', search_query.strip())
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
    print(f"\n[INFO] Searching for: {garment_data['product_hint']}")
    print(f"[URL] {url}")

    driver.get(url)
    time.sleep(3)
    wait = WebDriverWait(driver, 10)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-component-type='s-search-result']")))

    product_containers = driver.find_elements(By.CSS_SELECTOR, "[data-component-type='s-search-result']")
    print(f"[FOUND] {len(product_containers)} products")

    results = []
    for i, container in enumerate(product_containers[:max_products]):
        product_data = scrape_product_data(driver, container, i + 1)
        if product_data:
            product_data['search_parameters'] = garment_data
            product_data['product_index'] = i + 1
            results.append(product_data)
        time.sleep(1)

    return results

def scrape_multi_garment(garments, output_file, max_products=7):
    # If garments is a filename, load it; if it's a list, use as is
    if isinstance(garments, str):
        garments = read_multi_input(garments)
    if not garments:
        return

    all_results = []
    driver = setup_driver()

    try:
        for garment_data in garments:
            garment_results = scrape_for_each_garment(driver, garment_data, max_products=max_products)
            all_results.extend(garment_results)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"\n[SUCCESS] Total {len(all_results)} products scraped across {len(garments)} garments")
        print(f"[OUTPUT] Output saved to: {output_file}")

    except TimeoutException:
        print("[ERROR] Timeout waiting for page elements")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
    finally:
        driver.quit()
        print("[CLOSED] Browser closed")

def llm_recs_to_garment_inputs(llm_recs, occasion="casual"):
    """
    Convert LLM recommendations to garment input format for scraping.
    Now works with the simplified output format (Product Name, Color Palette, Fit)
    """
    garment_inputs = []
    
    for rec in llm_recs:
        # Extract color from color palette (take first color mentioned)
        color = "N/A"
        if 'Color Palette' in rec:
            # Get the first color mentioned in the palette
            colors = [c.strip() for c in rec['Color Palette'].split(',')]
            if colors:
                color = colors[0].split()[0]  # Take first word of first color
        
        garment_input = {
            "product_hint": rec.get('Product Name', ''),
            "color": color,
            "fit": rec.get('Fit', ''),
            "occasion": occasion  # Use the provided occasion
        }
        garment_inputs.append(garment_input)
    
    return garment_inputs

def scrape_from_llm_recommendations(llm_json_file, output_file, max_products=7, occasion="casual"):
    # Read LLM recommendations
    try:
        with open(llm_json_file, 'r', encoding='utf-8') as f:
            llm_recs = json.load(f)
    except Exception as e:
        print(f"Error reading {llm_json_file}: {e}")
        return

    # Convert to garment inputs with the provided occasion
    garment_inputs = llm_recs_to_garment_inputs(llm_recs, occasion)
    
    if not garment_inputs:
        print("No valid garments to scrape.")
        return
        
    scrape_multi_garment(garment_inputs, output_file, max_products=max_products)

def create_sample_scraped_data():
    """Create sample scraped data for testing when real scraping fails"""
    sample_products = [
        {
            "brand": "Nike",
            "description": "Men's Cotton Casual T-Shirt - Navy Blue",
            "price": "₹899",
            "rating": "4.2 out of 5 stars",
            "number_of_reviews": "1,234",
            "image_link": "https://m.media-amazon.com/images/I/71HblAhdXxL._UX679_.jpg",
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
            "image_link": "https://m.media-amazon.com/images/I/61vFO3ijCeL._UX679_.jpg",
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
            "image_link": "https://m.media-amazon.com/images/I/71YGQ5X8NFL._UX679_.jpg",
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
    occasion = "diwali"  # You can get this from user input or command line args
    scrape_from_llm_recommendations(input_file, output_file, max_products=7, occasion=occasion)
