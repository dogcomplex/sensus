import requests
import numpy as np
import argparse
import os
import time # Added for retry loop

ANU_API_URL = "https://qrng.anu.edu.au/API/jsonI.php"

def fetch_anu_randomness(n_numbers, data_type='uint16', block_size=1024):
    """
    Fetches high-quality random numbers from the ANU Quantum Random Number Generator.
    
    Args:
        n_numbers (int): The total number of random numbers to fetch.
        data_type (str): The data type to request ('uint16' or 'hex16').
        block_size (int): The number of values to fetch per API call (max 1024).

    Returns:
        np.array: An array of random numbers.
    """
    print(f"Fetching {n_numbers} quantum random numbers from ANU...")
    random_numbers = []
    remaining = n_numbers
    retries = 3

    while remaining > 0:
        fetch_count = min(remaining, block_size)
        params = {'length': fetch_count, 'type': data_type, 'size': 1} # size=1 is ignored for this API but good practice
        
        try:
            response = requests.get(ANU_API_URL, params=params, timeout=60) # Increased timeout
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            
            if not data['success']:
                raise ConnectionError(f"ANU API reported failure: {data}")

            random_numbers.extend(data['data'])
            remaining -= fetch_count
            retries = 3 # Reset retries on success
            if remaining > 0:
                print(f"  ... {len(random_numbers)}/{n_numbers} fetched.")

        except requests.exceptions.RequestException as e:
            retries -= 1
            print(f"Error fetching data from ANU API: {e}. Retries left: {retries}")
            if retries <= 0:
                print("Max retries reached. Aborting fetch from ANU.")
                return None
            time.sleep(5) # Wait 5 seconds before retrying
            
    print("Fetch complete.")
    return np.array(random_numbers, dtype=np.uint16)

def generate_chsh_settings_file(output_path, n_steps):
    """
    Generates a CHSH settings file (a, b) from the fetched randomness.
    We only need one bit per setting, so we can get many settings from one number.
    """
    # Each uint16 gives us 16 bits. We need 2 bits per step (one for a, one for b).
    # So we need n_steps / 8 numbers. Add buffer.
    n_to_fetch = int(np.ceil(n_steps / 8)) + 1 
    
    random_data = fetch_anu_randomness(n_to_fetch)
    if random_data is None:
        print("WARNING: Falling back to pseudo-random number generator (numpy.random).")
        print("         The generated file will not be from a true quantum source.")
        rng = np.random.default_rng()
        # Generate enough bytes for the bit stream directly
        n_bytes_needed = int(np.ceil(n_steps * 2 / 8))
        random_data = rng.bytes(n_bytes_needed)
        # Convert bytes to numpy array of uint8 for unpackbits
        random_data = np.frombuffer(random_data, dtype=np.uint8)

    # Unpack the uint16/uint8 numbers into a stream of bits
    bit_stream = np.unpackbits(random_data.view(np.uint8))
    
    if len(bit_stream) < n_steps * 2:
        print("Error: Did not fetch enough random bits.")
        return
        
    # Create the two columns for a_settings and b_settings
    a_settings = bit_stream[0:n_steps]
    b_settings = bit_stream[n_steps:n_steps*2]
    
    settings_matrix = np.vstack((a_settings, b_settings)).T
    
    # Save to file
    np.savetxt(output_path, settings_matrix, fmt='%d')
    print(f"Successfully saved {n_steps} CHSH settings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Fetch true random numbers from ANU and create a CHSH settings file.")
    parser.add_argument(
        '--steps',
        type=int,
        required=True,
        help="The number of simulation steps (i.e., rows of settings) to generate."
    )
    parser.add_argument(
        '--output',
        type=str,
        default='apsu/random_settings.txt',
        help="Path to save the output settings file."
    )
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    generate_chsh_settings_file(args.output, args.steps)

if __name__ == "__main__":
    main() 