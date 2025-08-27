import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from mt5_trading_bot.broker_interface import MT5Broker as Broker
except Exception as e:  # pragma: no cover - fallback when MT5Broker isn't available
    from mt5_trading_bot.broker_interface import MockBroker as Broker
    logger.warning(f"Falling back to MockBroker: {e}")


def monitor_bot():
    broker = Broker()
    while True:
        positions = broker.get_open_positions()
        for pos in positions:
            logger.info(
                f"Position: {pos.ticket}, Type: {'Buy' if pos.type == 1 else 'Sell'}, "
                f"Price: {pos.price_open}, SL: {pos.sl}"
            )
        time.sleep(300)  # Check every 5 minutes
    broker.close()


if __name__ == "__main__":
    monitor_bot()
