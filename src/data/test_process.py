"""
test_process.py
"""
import os
import argparse
import logging

from src.logger_config import logger_setup
logger_setup()
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def preprocessor(data):
    """
    Preprocess input data.
    
    Args:
        data (str): Input data.
    
    Returns:
        str: uppercase version of input data.
    """
    return data.upper()


def process_input(input_dir, output_dir, filename):
    """
    Process input data and save to output directory.
    
    Args:
        input_dir (str): Directory containing input data.
        output_dir (str): Directory to save processed data.
        filename (str): Name of file to process.
    """
    
    input_file = os.path.join(input_dir, filename)
    output_file = os.path.join(output_dir, filename)

    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} does not exist.")
        return
    
    with open(input_file, "r") as f:
        data = f.read()

    # perform processing
    processed_data = preprocessor(data)

    os.makedirs(output_dir, exist_ok=True)
    with open(output_file, "w") as f:
        f.write(processed_data)
    logger.info(f"Processed data saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sagemaker processing script")
    parser.add_argument("--input_dir", type=str, default="input", help="Directory containing input data")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save processed data")
    parser.add_argument("--filename", type=str, default="input.txt", help="Name of file to process")

    args = parser.parse_args()
    process_input(args.input_dir, args.output_dir, args.filename)