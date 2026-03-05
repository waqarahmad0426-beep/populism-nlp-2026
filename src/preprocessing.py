import os
import glob
import json
import pandas as pd
from preprocessing import preprocess_dataframe

def main():
    raw_dir = os.path.join("data", "raw", "miller_speeches", "speeches")
    json_files = glob.glob(os.path.join(raw_dir, "*.json"))
    
    data = []
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as file:
            try:
                content = json.load(file)
                date_str = content.get("date", "")
                year = date_str[:4] if date_str and len(date_str) >= 4 else "Unknown"
                
                data.append({
                    "president": content.get("president", "Unknown"),
                    "title": content.get("title", ""),
                    "date": date_str,
                    "year": year,
                    "text": content.get("transcript", "")
                })
            except Exception as e:
                print(f"Error reading {f}: {e}")
                
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} speeches.")
    
    # We might want to sample if taking too long, but let's process all first
    # Process dataframe
    df_clean = preprocess_dataframe(df, text_col='text')
    
    # Save to CSV
    out_dir = os.path.join("data", "cleaned")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cleaned_speeches.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
