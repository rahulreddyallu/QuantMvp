"""
Enhanced Quantitative Analysis Engine Main Entry Point
Runs the trading bot based on configuration settings

Version: 3.5.1
Author: rahulreddyallu
Date: 2025-05-01
"""

import os
import sys
import argparse
import asyncio
import logging
import datetime
import traceback

# Import configuration
from config import get_config

# Import trading bot implementation
from compute import (
    setup_logging, 
    main as compute_main, 
    test_upstox_connection, 
    test_telegram_connection,
    send_startup_notification
)

async def validate_and_test_connections(config, logger):
    """
    Validate configuration and test API connections
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (config_valid, upstox_ok, telegram_ok)
    """
    # Check required configuration
    config_valid = True
    
    # Validate Upstox API credentials
    if not config.get('UPSTOX_ACCESS_TOKEN'):
        logger.error("UPSTOX_ACCESS_TOKEN is required but not configured")
        config_valid = False
    
    # Check STOCK_LIST
    if not config.get('STOCK_LIST'):
        logger.error("STOCK_LIST is empty - no stocks configured for analysis")
        config_valid = False
    
    # Test Upstox connection
    upstox_ok = test_upstox_connection(config, logger)
    if not upstox_ok:
        logger.error("Failed to connect to Upstox API - please check your credentials")
    
    # Test Telegram connection if enabled
    telegram_ok = False
    if config.get('ENABLE_TELEGRAM_ALERTS', False):
        if not config.get('TELEGRAM_BOT_TOKEN') or not config.get('TELEGRAM_CHAT_ID'):
            logger.error("Telegram alerts are enabled but credentials are missing")
            config_valid = False
        else:
            telegram_ok = await test_telegram_connection(config, logger)
            if not telegram_ok:
                logger.warning("Telegram connection failed - notifications will not be sent")
    
    return config_valid, upstox_ok, telegram_ok

async def async_main(args):
    """
    Main async function to run the Trading Bot
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Get configuration
        config = get_config()
        
        # Override with command line arguments if provided
        if args.access_token:
            config['UPSTOX_ACCESS_TOKEN'] = args.access_token
        
        if args.telegram_token:
            config['TELEGRAM_BOT_TOKEN'] = args.telegram_token
            
        if args.telegram_chat_id:
            config['TELEGRAM_CHAT_ID'] = args.telegram_chat_id
            
        if args.log_dir:
            config['LOG_DIRECTORY'] = args.log_dir
            
        if args.disable_telegram:
            config['ENABLE_TELEGRAM_ALERTS'] = False
            
        if args.run_once:
            config['SCHEDULED_MODE'] = False
            config['RUN_ON_STARTUP'] = True
        
        # Setup logging
        logger = setup_logging(config)
        
        logger.info("="*70)
        logger.info(f"Quantitative Trading Bot v{config['VERSION']} - Starting")
        logger.info(f"Bot run by: rahulreddyallu at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("="*70)
        
        # Print list of stocks being analyzed
        stock_count = len(config.get('STOCK_LIST', []))
        logger.info(f"Configured to analyze {stock_count} stocks with {config.get('HISTORICAL_DAYS', 100)} days of historical data")
        
        # Validate configuration and test connections
        config_valid, upstox_ok, telegram_ok = await validate_and_test_connections(config, logger)
        
        if not config_valid:
            logger.error("Invalid configuration - please fix the issues above")
            return 1
        
        if not upstox_ok:
            logger.error("Cannot proceed without Upstox API connection")
            return 1
        
        # Send startup notification
        if config.get('ENABLE_TELEGRAM_ALERTS', False) and telegram_ok:
            await send_startup_notification(config, logger)
        
        # Run the main compute function
        exit_code = compute_main(config)
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

def main():
    """
    Main entry point for the Trading Bot
    Parses command line arguments and runs the bot
    """
    parser = argparse.ArgumentParser(description='Enhanced Quantitative Trading Bot')
    
    parser.add_argument('--access-token', dest='access_token', 
                        help='Upstox API access token')
    
    parser.add_argument('--telegram-token', dest='telegram_token',
                        help='Telegram Bot API token')
    
    parser.add_argument('--telegram-chat-id', dest='telegram_chat_id',
                        help='Telegram chat ID for notifications')
    
    parser.add_argument('--log-dir', dest='log_dir',
                        help='Directory for log files')
    
    parser.add_argument('--disable-telegram', dest='disable_telegram', 
                        action='store_true', help='Disable Telegram notifications')
    
    parser.add_argument('--run-once', dest='run_once', action='store_true',
                        help='Run analysis once and exit (disable scheduling)')
    
    parser.add_argument('--version', action='version', version='%(prog)s 3.5.1')
    
    args = parser.parse_args()
    
    # Run the async main function
    exit_code = asyncio.run(async_main(args))
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
