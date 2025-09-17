"""
Bot service runner for local testing and development.
"""

import sys
import argparse
import asyncio
from pathlib import Path
from aiohttp import web

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.azure_qlora_healthcare.deployment.bot_service import create_bot_app
from src.azure_qlora_healthcare.utils.logger import get_logger

def main():
    """Run the healthcare bot service."""
    parser = argparse.ArgumentParser(description="Healthcare Bot Service")
    parser.add_argument("--model-path", type=str, help="Path to trained model")
    parser.add_argument("--port", type=int, default=3978, help="Port to run the bot service")
    parser.add_argument("--host", type=str, default="localhost", help="Host to run the bot service")
    
    args = parser.parse_args()
    
    logger = get_logger(__name__)
    logger.info("Starting Healthcare Bot Service...")
    
    try:
        # Create bot application
        app = create_bot_app()
        
        # Run the application
        logger.info(f"Bot service starting on http://{args.host}:{args.port}")
        logger.info("Bot endpoint: /api/messages")
        
        web.run_app(app, host=args.host, port=args.port)
        
    except Exception as e:
        logger.error(f"Failed to start bot service: {e}")
        raise

if __name__ == "__main__":
    main()