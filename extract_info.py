import re

def parse_llm_recommendations(text_output):
    """
    Parses the plain text output from the LLM into a structured list of dictionaries,
    where each dictionary represents a clothing style recommendation.
    """
    recommendations = []
    # Split the output into individual style blocks
    # Use re.DOTALL to ensure '.' matches newlines as well, for multi-line blocks
    style_blocks = re.split(r'\n\s*\d+\.\s*Style Name:', text_output, flags=re.DOTALL)

    for block in style_blocks:
        if not block.strip():
            continue # Skip empty blocks

        style_data = {}
        # Extract Style Name (first line of the block, after splitting)
        # We need to handle cases where the split might leave "Style Name: " at the start
        first_line = block.strip().split('\n')[0]
        style_name_match = re.match(r'(?:Style Name:\s*)?(.+)', first_line) # (?:...) is a non-capturing group
        if style_name_match:
            style_data['Style Name'] = style_name_match.group(1).strip()


        # Regex patterns for each attribute
        # Using re.DOTALL for patterns that might span multiple lines if necessary
        patterns = {
            'Garment Type': r'- Garment Type: (.+)',
            'Color Palette': r'- Color Palette: (.+)',
            'Fit': r'- Fit: (.+)',
            'Fabric': r'- Fabric: (.+)',
            'Accessories': r'- Accessories: (.+)',
            'Product Name': r'- Product Name: (.+)'
        }

        # Iterate through the rest of the block lines to find attributes
        for key, pattern in patterns.items():
            match = re.search(pattern, block)
            if match:
                style_data[key] = match.group(1).strip()
            else:
                style_data[key] = None # Or an empty string if preferred

        if style_data and style_data.get('Style Name'): # Only add if we successfully extracted a style name
            recommendations.append(style_data)

    return recommendations

