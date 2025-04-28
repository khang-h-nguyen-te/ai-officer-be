"""
Utilities for running the application in serverless environments like Vercel.
These functions handle common issues with read-only filesystems and other constraints.
"""

import os
import logging
import tempfile

logger = logging.getLogger(__name__)

def configure_for_serverless():
    """
    Configure the application for running in serverless environments.
    
    This handles:
    - Setting up tiktoken to use a writable directory
    - Other serverless-specific configurations
    
    Returns:
        bool: True if configuration was successful
    """
    try:
        # Create temp directory for tiktoken cache
        tmp_dir = tempfile.mkdtemp()
        os.environ["TIKTOKEN_CACHE_DIR"] = tmp_dir
        logger.info(f"Set TIKTOKEN_CACHE_DIR to {tmp_dir}")
        
        # Verify the directory is writable
        test_file = os.path.join(tmp_dir, "test_write.txt")
        try:
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info("Verified temp directory is writable")
        except Exception as e:
            logger.warning(f"Temp directory is not writable: {e}")
            # If tmp_dir is not writable, try /tmp which is usually writable in serverless
            os.environ["TIKTOKEN_CACHE_DIR"] = "/tmp/tiktoken_cache"
            os.makedirs("/tmp/tiktoken_cache", exist_ok=True)
            logger.info("Falling back to /tmp/tiktoken_cache")
        
        # Configure any other serverless-specific settings here
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure for serverless: {e}")
        return False 