import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, Optional

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()

class ETFTrader:
    def __init__(self):
        # Component ratios
        self.ratios = {
            "CHOCOLATE": 4,
            "STRAWBERRIES": 6,
            "ROSES": 1
        }
        
        # Position limits
        self.position_limits = {
            "GIFT_BASKET": 50,
            "CHOCOLATE": 200,
            "STRAWBERRIES": 300,
            "ROSES": 50
        }
        
        # Thresholds for trading
        self.open_threshold = 50
        self.close_threshold = 0

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculate mid price from order book"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        return (best_bid + best_ask) / 2
    
    def calculate_residual(self, mid_prices: Dict[str, float]) -> Optional[float]:
        """Calculate residual between GIFT_BASKET and component prices"""
        # Check if we have all required prices
        required_symbols = ["GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES", "ROSES"]
        if not all(symbol in mid_prices for symbol in required_symbols):
            return None
        
        # Calculate theoretical fair value of GIFT_BASKET
        theoretical_value = (
            self.ratios["CHOCOLATE"] * mid_prices["CHOCOLATE"] +
            self.ratios["STRAWBERRIES"] * mid_prices["STRAWBERRIES"] +
            self.ratios["ROSES"] * mid_prices["ROSES"]
        )
        
        # Calculate residual
        return mid_prices["GIFT_BASKET"] - theoretical_value
    
    def get_max_trade_size(self, current_positions: Dict[str, int]) -> int:
        """Calculate maximum trade size based on position limits"""
        # Calculate how much more we can trade for each asset
        remaining_capacity = {}
        for symbol, limit in self.position_limits.items():
            current_pos = current_positions.get(symbol, 0)
            remaining_long = limit - current_pos
            remaining_short = limit + current_pos
            # The most constraining direction determines our trade size
            remaining_capacity[symbol] = min(remaining_long, remaining_short)
        
        # Now adjust for the ratios
        max_basket_size = remaining_capacity["GIFT_BASKET"]
        max_chocolate_size = remaining_capacity["CHOCOLATE"] // self.ratios["CHOCOLATE"]
        max_strawberries_size = remaining_capacity["STRAWBERRIES"] // self.ratios["STRAWBERRIES"]
        max_roses_size = remaining_capacity["ROSES"] // self.ratios["ROSES"]
        
        # The most constraining component determines our overall trade size
        return min(max_basket_size, max_chocolate_size, max_strawberries_size, max_roses_size)
    
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Trading strategy for ETF vs. components arbitrage:
        - Calculate residual between GIFT_BASKET and components
        - Trade when residual exceeds thresholds
        """
        result = {}
        
        # Check if we have all required symbols
        required_symbols = ["GIFT_BASKET", "CHOCOLATE", "STRAWBERRIES", "ROSES"]
        if not all(symbol in state.order_depths for symbol in required_symbols):
            logger.flush(state, result, 0, "")
            return result, 0, ""
        
        # Get current positions
        current_positions = {
            symbol: state.position.get(symbol, 0) 
            for symbol in required_symbols
        }
        
        # Calculate mid prices for all assets
        mid_prices = {}
        for symbol in required_symbols:
            mid_price = self.get_mid_price(state.order_depths[symbol])
            if mid_price:
                mid_prices[symbol] = mid_price
        
        # Calculate residual
        residual = self.calculate_residual(mid_prices)
        if residual is None:
            logger.flush(state, result, 0, "")
            return result, 0, ""
        
        # Initialize orders dict
        for symbol in required_symbols:
            result[symbol] = []
        
        # Determine if we should open a new position or close an existing one
        # Check if we currently have a position
        has_position = current_positions["GIFT_BASKET"] != 0
        
        if not has_position:
            # CASE 1: Opening a new position
            if residual > self.open_threshold:
                # Residual is positive and large - short the basket, long the components
                max_size = self.get_max_trade_size(current_positions)
                if max_size > 0:
                    # Short GIFT_BASKET
                    if "GIFT_BASKET" in mid_prices:
                        result["GIFT_BASKET"].append(Order("GIFT_BASKET", mid_prices["GIFT_BASKET"], -max_size))
                    
                    # Long components
                    for component, ratio in self.ratios.items():
                        if component in mid_prices:
                            component_size = max_size * ratio
                            result[component].append(Order(component, mid_prices[component], component_size))
            
            elif residual < -self.open_threshold:
                # Residual is negative and large - long the basket, short the components
                max_size = self.get_max_trade_size(current_positions)
                if max_size > 0:
                    # Long GIFT_BASKET
                    if "GIFT_BASKET" in mid_prices:
                        result["GIFT_BASKET"].append(Order("GIFT_BASKET", mid_prices["GIFT_BASKET"], max_size))
                    
                    # Short components
                    for component, ratio in self.ratios.items():
                        if component in mid_prices:
                            component_size = max_size * ratio
                            result[component].append(Order(component, mid_prices[component], -component_size))
        
        else:
            # CASE 2: We have an existing position - check if we should close it
            if abs(residual) < self.close_threshold:
                # Residual is close to zero - close positions
                
                # Close GIFT_BASKET position
                basket_pos = current_positions["GIFT_BASKET"]
                if basket_pos != 0 and "GIFT_BASKET" in mid_prices:
                    result["GIFT_BASKET"].append(Order("GIFT_BASKET", mid_prices["GIFT_BASKET"], -basket_pos))
                
                # Close component positions
                for component in self.ratios:
                    component_pos = current_positions.get(component, 0)
                    if component_pos != 0 and component in mid_prices:
                        result[component].append(Order(component, mid_prices[component], -component_pos))
            
            # If residual hasn't reached close threshold, hold the position
        
        logger.flush(state, result, 0, "")
        return result, 0, ""

def Trader():
    """Factory function that returns a trader instance"""
    return ETFTrader()