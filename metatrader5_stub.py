META_TRADER5_STUB = True

class SymbolInfo:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.point = 0.0001
        self.trade_contract_size = 100000

class SymbolInfoTick:
    def __init__(self):
        self.ask = 1.0
        self.bid = 1.0

class OrderSendResult:
    def __init__(self):
        self.retcode = TRADE_RETCODE_DONE
        self.comment = ""

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_ACTION_CLOSE_BY = 2
TRADE_ACTION_MODIFY = 6
TRADE_RETCODE_DONE = 0
ORDER_TIME_GTC = 1
ORDER_FILLING_IOC = 1


def initialize(*args, **kwargs):
    return True


def shutdown():
    return True


def login(*args, **kwargs):
    return True


def symbol_select(symbol, enable):
    return True


def symbol_info(symbol):
    return SymbolInfo(symbol)


def symbol_info_tick(symbol):
    return SymbolInfoTick()


def copy_rates_from_pos(symbol, timeframe, start_pos, count):
    return []


def copy_rates_range(symbol, timeframe, from_date, to_date):
    return []


def positions_get(*args, **kwargs):
    return []


def order_send(request):
    return OrderSendResult()


def last_error():
    return (0, "stub")
