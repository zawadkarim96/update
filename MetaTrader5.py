class SymbolInfo:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.point = 0.0001
        self.trade_contract_size = 100000

class OrderSendResult:
    def __init__(self):
        self.retcode = TRADE_RETCODE_DONE
        self.comment = ""

TRADE_RETCODE_DONE = 0
TRADE_ACTION_MODIFY = 6
ORDER_TIME_GTC = 1
ORDER_FILLING_IOC = 1

def initialize(*args, **kwargs):
    return True

def shutdown():
    return True

def login(*args, **kwargs):
    return True

def symbol_info(symbol):
    return SymbolInfo(symbol)

def positions_get(*args, **kwargs):
    return []

def order_send(request):
    return OrderSendResult()
