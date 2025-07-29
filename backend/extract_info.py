import re
import json

def parse_llm_recommendations(text):
    recommendations = []
    
    blocks = re.split(r'---\s*Recommendation\s*\d+\s*---', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        recommendation = {}

        # Only these 3 fields
        for field in ["Product Name", "Fit", "Color Palette", "Gender"]:
            match = re.search(rf"{field}\s*:\s*(.+)", block)
            if match:
                recommendation[field] = match.group(1).strip()

        # Only append if all 3 are present
        if len(recommendation) == 4:
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