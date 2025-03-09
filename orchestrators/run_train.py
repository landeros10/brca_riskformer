'''
run_sagemaker_preprocess.py

Run a preprocessing job on SageMaker.
Author: landeros10
Created: 2025-02-05
'''
import os
import logging
import argparse
import numpy as np

import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

from riskformer.utils.logger_config import logger_setup
from riskformer.utils.data_utils import initialize_s3_client, load_slide_paths
logger = logging.getLogger(__name__)


def main():
    # set up arg parsing
    parser = argparse.ArgumentParser(description="Data loading script")

if __name__ == "__main__":
    main()