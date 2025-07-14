import requests
import numpy as np
import os
import argparse
import logging
import getpass
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
API_URL = "https://api.quantumnumbers.anu.edu.au/"
MAX_LENGTH_PER_REQUEST = 1024

def fetch_quantum_randomness(n_bytes, api_key):
    """
    Fetches a specified number of random bytes from the ANU Quantum Numbers API.
    
    Args:
        n_bytes (int): The total number of random bytes to fetch.
        api_key (str): The API key for the ANU service.
        
    Returns:
        np.ndarray: An array of random bytes, or None if the fetch fails.
    """
    if not api_key:
        logging.error("API key is missing.")
        return None

    headers = {"x-api-key": api_key}
    random_bytes = []
    
    logging.info(f"Fetching {n_bytes} bytes from ANU Quantum Random Numbers API...")
    
    for i in range(0, n_bytes, MAX_LENGTH_PER_REQUEST):
        bytes_to_fetch = min(MAX_LENGTH_PER_REQUEST, n_bytes - i)
        params = {"length": bytes_to_fetch, "type": "uint8"}
        
        try:
            response = requests.get(API_URL, headers=headers, params=params, timeout=10)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            js = response.json()
            if js.get("success"):
                random_bytes.extend(js["data"])
            else:
                logging.error(f"API returned an error: {js.get('message', 'Unknown error')}")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred during API request: {e}")
            return None
            
    logging.info(f"Successfully fetched {len(random_bytes)} bytes.")
    return np.array(random_bytes, dtype=np.uint8)

def main():
    """Main function to fetch and save randomness."""
    parser = argparse.ArgumentParser(description="Fetch true quantum randomness from ANU API.")
    parser.add_argument(
        "--bytes",
        type=int,
        default=16384,
        help="Number of random bytes to fetch."
    )
    parser.add_argument(
        "--out",
        type=str,
        default="apsu/experiments/qrng_chsh_settings.bin",
        help="Output file path to save the random bytes."
    )
    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment variable or prompt the user
    api_key = os.environ.get("AQN_API_KEY")
    if not api_key:
        logging.warning("AQN_API_KEY environment variable not set.")
        try:
            api_key = getpass.getpass("Please enter your ANU QRNG API key: ")
        except (EOFError, KeyboardInterrupt):
            logging.error("\nCould not read API key. Aborting.")
            return

    random_data = fetch_quantum_randomness(args.bytes, api_key)
    
    if random_data is not None:
        # Ensure the output directory exists
        output_dir = os.path.dirname(args.out)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the data
        with open(args.out, 'wb') as f:
            f.write(random_data.tobytes())
        logging.info(f"Saved {len(random_data)} random bytes to {args.out}")

if __name__ == "__main__":
    main() 