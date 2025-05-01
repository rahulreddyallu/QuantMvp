"""
Quantitative Trading Signal Generator
Simple execution starter for the Enhanced Trading Signal Bot

Current Date and Time (UTC): 2025-05-01 10:58:02
Current User's Login: rahulreddyallu
"""

import os
import sys
import logging
import asyncio
import datetime
import traceback
from typing import Dict, Any, Optional

# Import configuration and compute module
try:
    from config import *
    import compute
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure config.py and compute.py are in the same directory as main.py")
    sys.exit(1)

def setup_logging() -> logging.Logger:
    """
    Configure the logging system
    
    Returns:
        Logger object configured for console and file output
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate log filename with current date
    log_filename = f"logs/trading_bot_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    
    # Configure logging to file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("trading_bot")

async def run_analysis(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """
    Run the signal generation analysis once
    
    Args:
        config: Configuration dictionary
        logger: Logger object
    
    Returns:
        Boolean indicating success or failure
    """
    logger.info("=" * 80)
    logger.info(f"STARTING ANALYSIS - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    logger.info(f"Version: {config.get('VERSION', '3.5.0')}")
    logger.info(f"Analyzing {len(config.get('STOCK_LIST', []))} symbols")
    logger.info("=" * 80)
    
    try:
        # Initialize and test connections
        upstox_ok, telegram_ok = await compute.initialize_and_test(config)
        
        if not upstox_ok:
            logger.error("Cannot proceed without Upstox API connection")
            return False
            
        if not telegram_ok and config.get('ENABLE_TELEGRAM_ALERTS', False):
            logger.warning("Telegram connection failed, proceeding without notifications")
        
        # Send startup notification if enabled
        if config.get('ENABLE_TELEGRAM_ALERTS', False):
            await compute.send_startup_notification(config)
        
        # Run the actual analysis
        result = await compute.run_trading_signals(config)
        
        logger.info("=" * 80)
        logger.info(f"ANALYSIS COMPLETED - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def get_config() -> Dict[str, Any]:
    """
    Get configuration from config.py
    
    Returns:
        Dictionary with configuration values
    """
    config = {}
    
    # Import all uppercase variables from config module
    for var in dir():
        if var.isupper():
            config[var] = globals()[var]
            
    # Provide some hardcoded defaults if not found
    if 'VERSION' not in config:
        config['VERSION'] = "3.5.0"
    
    if 'HISTORICAL_DAYS' not in config:
        config['HISTORICAL_DAYS'] = 100
        
    if 'CHART_INTERVAL' not in config:
        config['CHART_INTERVAL'] = "day"
    
    if 'MINIMUM_SIGNAL_STRENGTH' not in config:
        config['MINIMUM_SIGNAL_STRENGTH'] = 3
    
    return config

def main() -> int:
    """
    Main function to run the Trading Signal Bot once
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Set up logging
    logger = setup_logging()
    
    try:
        # Get configuration
        config = get_config()
        
        # Print welcome message
        print(f"\n{'=' * 80}")
        print(f"  Enhanced Quantitative Trading Signal Generator v{config.get('VERSION', '3.5.0')}")
        print(f"  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  Analyzing {len(config.get('STOCK_LIST', []))} symbols with {config.get('HISTORICAL_DAYS', 100)} days of history")
        print(f"  Author: rahulreddyallu")
        print(f"{'=' * 80}\n")
        
        # Create or get event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the analysis
        result = loop.run_until_complete(run_analysis(config, logger))
        
        if result:
            print("\n✅ Analysis completed successfully!")
            return 0
        else:
            print("\n❌ Analysis failed. Check the logs for details.")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        return 0
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
