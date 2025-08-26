from mt5_trading_bot.main import main
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_demo_test():
    logger.info("Starting demo test at 2025-08-25 02:51 PM +06")
    main(live_mode=True, active_strategies=['moving_average_crossover'])
    logger.info("Demo test completed")

if __name__ == "__main__":
    run_demo_test()