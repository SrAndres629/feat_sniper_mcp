import json
import sys
import argparse

INDEX_FILE = "project_atlas.json"

def search_index(query, index_data):
    results = []
    print(f"🔍 Querying Atlas for: '{query}'...")
    
    for filename, content in index_data.items():
        # Search Classes
        for class_name, details in content.get('classes', {}).items():
            if query.lower() in class_name.lower():
                 results.append(f"[CLASS] {class_name} in {filename}")
            # Search Methods in Class
            for method in details.get('methods', []):
                if query.lower() in method.lower():
                    results.append(f"[METHOD] {class_name}.{method} in {filename}")

        # Search Functions
        for func in content.get('functions', []):
             if query.lower() in func['name'].lower():
                results.append(f"[FUNC] {func['name']} in {filename}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: nexus_query.py <term>")
        sys.exit(1)
        
    query = sys.argv[1]
    
    try:
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            hits = search_index(query, data)
            
            if hits:
                print("\n".join(hits))
            else:
                print("No neural matches found.")
    except Exception as e:
        print(f"Cortex Access Error: {e}")
