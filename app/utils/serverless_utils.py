"""
Utilities for running the application in serverless environments like Vercel.
These functions handle common issues with read-only filesystems and other constraints.
"""

import os
import logging
import tempfile
import json

logger = logging.getLogger(__name__)

def configure_for_serverless():
    """
    Configure the application for running in serverless environments.
    
    This handles:
    - Setting up tiktoken to use a writable directory
    - Configuring timeouts for API clients
    - Setting up environment detection
    
    Returns:
        bool: True if configuration was successful
    """
    try:
        # Detect serverless environment
        is_vercel = os.environ.get("VERCEL") == "1"
        is_aws_lambda = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
        is_serverless = is_vercel or is_aws_lambda
        
        if is_serverless:
            logger.info(f"Detected serverless environment: Vercel={is_vercel}, AWS Lambda={is_aws_lambda}")
        
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
        
        # Create indicator file in /tmp to help with debugging
        if is_serverless:
            try:
                env_info = {
                    "is_vercel": is_vercel,
                    "is_aws_lambda": is_aws_lambda,
                    "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
                    "startup_time": os.environ.get("NOW_READY_TIME", "unknown"),
                    "region": os.environ.get("VERCEL_REGION", "unknown")
                }
                
                with open("/tmp/serverless_info.json", "w") as f:
                    json.dump(env_info, f)
                
                logger.info("Created serverless environment information file")
            except Exception as e:
                logger.warning(f"Failed to create serverless info file: {e}")
        
        # Configure faster timeouts for serverless
        if is_serverless:
            # Set default timeouts for common libraries
            os.environ["HTTPX_TIMEOUT"] = "15"  # 15 seconds for HTTP requests
            os.environ["OPENAI_TIMEOUT"] = "20"  # 20 seconds for OpenAI API
            
            # Reduce memory usage for serverless
            os.environ["MEMORY_TOKEN_LIMIT"] = "5000"  # Smaller memory size
            
            logger.info("Configured serverless-specific timeouts and memory limits")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure for serverless: {e}")
        return False

def is_serverless_environment():
    """Check if the application is running in a serverless environment."""
    return os.environ.get("VERCEL") == "1" or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None

def get_serverless_info():
    """Get information about the serverless environment."""
    try:
        # Check if we already have the info file
        if os.path.exists("/tmp/serverless_info.json"):
            with open("/tmp/serverless_info.json", "r") as f:
                return json.load(f)
        
        # Otherwise gather information directly
        return {
            "is_vercel": os.environ.get("VERCEL") == "1",
            "is_aws_lambda": os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None,
            "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
            "memory_limit": os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "unknown"),
            "region": os.environ.get("VERCEL_REGION", "unknown")
        }
    except Exception as e:
        logger.error(f"Error getting serverless info: {e}")
        return {"error": str(e)} 