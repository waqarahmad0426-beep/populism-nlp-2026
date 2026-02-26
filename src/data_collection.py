import requests, tarfile, os
from tqdm import tqdm

def download_miller_center():
    url = "https://data.millercenter.org/miller_center_speeches.tgz"
    os.makedirs("data/raw", exist_ok=True)
    local_file = "data/raw/miller_center_speeches.tgz"
    
    if not os.path.exists(local_file):
        print("Downloading Miller Center speeches...")
        r = requests.get(url, stream=True)
        with open(local_file, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                f.write(chunk)
        print(" Download complete")
    
    with tarfile.open(local_file, 'r:gz') as tar:
        for member in tqdm(tar.getmembers(), desc="Extracting"):
            # Sanitize filename for Windows (replace invalid chars with _)
            # Windows invalid chars: < > : " / \ | ? *
            sanitized_name = member.name
            for char in '<>:"\\|?*': # / is path separator, so we keep it but handle the rest
                sanitized_name = sanitized_name.replace(char, '_')
            member.name = sanitized_name
            tar.extract(member, path="data/raw/miller_speeches")
    print(" Extracted successfully")

if __name__ == "__main__":
    download_miller_center()