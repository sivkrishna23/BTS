import requests
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_URL = "https://data.transportation.gov/api/views/keg4-3bc2/rows.csv?accessType=DOWNLOAD"
DESTINATION = Path("data/raw/Border_Crossing_Entry_Data.csv")

def fetch_data(url: str, destination: Path):
    """
    Downloads data from the specified URL to the destination path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting download from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Successfully downloaded data to {destination}")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_data(DATA_URL, DESTINATION)
