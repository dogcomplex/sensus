import os
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_anu_data(api_key, num_blocks=10):
    """Fetches multiple blocks of quantum random numbers from the ANU API."""
    base_url = "https://api.quantumnumbers.anu.edu.au"
    headers = {'x-api-key': api_key}
    all_data = []
    logging.info(f"Fetching {num_blocks} blocks of quantum randomness...")
    for i in range(num_blocks):
        # The 'size' parameter is for hex types, not needed for uint16
        params = {
            'length': 1024,
            'type': 'uint16'
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            if not data.get('success', False):
                logging.error(f"API Error on block {i+1}: {data.get('error', 'Unknown error')}")
                return None
            all_data.extend(data['data'])
            logging.info(f"Fetched block {i+1}/{num_blocks}...")
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP Request failed: {e}")
            return None
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON response. Check API status or key. Response: {response.text}")
            return None
    return np.array(all_data, dtype=np.uint16)

def main():
    """
    Main function to fetch randomness and split it into training and testing sets.
    """
    load_dotenv()
    api_key = os.getenv("ANU_API_KEY")

    if not api_key:
        logging.error("ANU_API_KEY not found in .env file.")
        logging.error("Please create a .env file in the project root with your API key:")
        logging.error("ANU_API_KEY=your_api_key_here")
        return

    output_dir = Path("apsu/utils")
    output_dir.mkdir(exist_ok=True)
    
    # We need enough data for two full simulation runs (4000 floats each)
    # 1 float32 = 4 bytes. 1 uint16 = 2 bytes. We need 4000 * 4 * 2 = 32000 bytes = 16000 uint16s.
    # Let's fetch more to be safe. 20 blocks * 1024 uint16s/block = 20480 uint16s.
    num_uint16_needed = 8000 * 2 # enough for 8000 floats
    num_blocks_to_fetch = int(np.ceil(num_uint16_needed / 1024)) + 1 # ~17 blocks
    
    random_data = fetch_anu_data(api_key, num_blocks=num_blocks_to_fetch)

    if random_data is None:
        logging.error("Failed to fetch random data. Aborting.")
        return

    # Ensure we have an even number of bytes for float32 conversion
    if len(random_data) % 2 != 0:
        random_data = random_data[:-1]

    # Convert to float32 in range [0, 2*pi]
    float_data = (random_data.astype(np.float32) / 65535.0) * 2 * np.pi

    # Split the data
    split_point = len(float_data) // 2
    training_data = float_data[:split_point]
    testing_data = float_data[split_point:]

    train_path = output_dir / "training_randomness.bin"
    test_path = output_dir / "testing_randomness.bin"

    with open(train_path, 'wb') as f:
        f.write(training_data.tobytes())
    logging.info(f"Saved {len(training_data)} floats to {train_path}")

    with open(test_path, 'wb') as f:
        f.write(testing_data.tobytes())
    logging.info(f"Saved {len(testing_data)} floats to {test_path}")
    logging.info("Double-blind data generation complete.")


if __name__ == "__main__":
    main() 