import re
import json

def parse_llm_recommendations(text):
    recommendations = []
    
    # Split by numbered recommendations (1., 2., 3., etc.)
    blocks = re.split(r'\n?\d+\.\s*Product Name:', text)
    
    for i, block in enumerate(blocks):
        if i == 0:  # Skip the first empty block before "1. Product Name:"
            continue
            
        block = block.strip()
        if not block:
            continue

        recommendation = {}
        
        # Extract Product Name (it's the first part after the split)
        product_name_match = re.match(r'^([^\n]+)', block)
        if product_name_match:
            recommendation["Product Name"] = product_name_match.group(1).strip()
        
        # Extract other fields
        for field in ["Fit", "Color Palette", "Gender"]:
            pattern = rf"{field}\s*:\s*([^\n]+)"
            match = re.search(pattern, block, re.IGNORECASE)
            if match:
                recommendation[field] = match.group(1).strip()

        # Only append if we have at least Product Name and 2 other fields
        if len(recommendation) >= 3 and "Product Name" in recommendation:
            recommendations.append(recommendation)

    return recommendations

def save_recommendations_to_json(recommendations, output_file):
    """Save recommendations to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

def save_facial_features_to_json(facial_features_dict, output_file):
    """Save the facial features dictionary to a JSON file for LLM input."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(facial_features_dict, f, indent=2, ensure_ascii=False)

def parse_llm_response_and_generate_final_output(llm_response):
    """Parse LLM response and generate final output format"""
    return parse_llm_recommendations(llm_response)

# Test
if __name__ == "__main__":
    gpt_output = """
    --- Recommendation 1 ---
    Product Name: Chic Linen Tee
    Fit: Slim
    Color Palette: White, Black
    Gender: Male

    --- Recommendation 2 ---
    Product Name: Relaxed Hoodie
    Fit: Loose
    Color Palette: Grey, Navy
    Gender: Female
    """

    result = parse_llm_recommendations(gpt_output)
    for i, rec in enumerate(result, 1):
        print(f"\n--- Recommendation {i} ---")
        print(f"Product Name: {rec['Product Name']}")
        print(f"Fit: {rec['Fit']}")
        print(f"Color Palette: {rec['Color Palette']}")
        print(f"Gender: {rec['Gender']}")