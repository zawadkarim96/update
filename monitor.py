import logging
from mt5_trading_bot.broker_interface import MT5Broker
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_bot():
    broker = MT5Broker()
    while True:
        positions = broker.get_open_positions()
        for pos in positions:
            logger.info(f"Position: {pos.ticket}, Type: {'Buy' if pos.type == 1 else 'Sell'}, "
                        f"Price: {pos.price_open}, SL: {pos.sl}")
        time.sleep(300)  # Check every 5 minutes
    broker.close()

if __name__ == "__main__":
    monitor_bot()