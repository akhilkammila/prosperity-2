import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List

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

class StarfruitTrader:
    def __init__(self):
        pass
    
    def find_best_prices_with_volume(self, order_depth: OrderDepth, min_volume: int = 15):
        """Find best bid and ask prices with minimum volume"""
        best_bid = None
        best_ask = None
        
        # Find best bid (highest buy price) with volume > min_volume
        if order_depth.buy_orders:
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if volume >= min_volume:
                    best_bid = price
                    break
        
        # Find best ask (lowest sell price) with volume > min_volume
        if order_depth.sell_orders:
            for price, volume in sorted(order_depth.sell_orders.items()):
                if abs(volume) >= min_volume:  # volume is negative in sell_orders
                    best_ask = price
                    break
        
        return best_bid, best_ask

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Trading strategy for STARFRUIT:
        - Calculate mid price based on best bid/ask with volume > 15
        - PART 1: Trade at favorable prices in the orderbook (threshold = 1)
        - PART 2: Try to clear position at mid price if possible
        - PART 3: Place resting orders at mid price ± 2
        """
        result = {}
        
        # Only trade STARFRUIT
        if "STARFRUIT" not in state.order_depths:
            return result, 0, ""

        # Get current position and order depth for STARFRUIT
        position = state.position.get("STARFRUIT", 0)
        order_depth = state.order_depths["STARFRUIT"]
        
        # Find best bid and ask with volume > 15
        best_bid, best_ask = self.find_best_prices_with_volume(order_depth)
        if best_bid is None: best_bid = best_ask - 7
        if best_ask is None: best_ask = best_bid + 7
            
        # Create orders list
        orders: list[Order] = []
        
        # Position limit
        position_limit = 20
        
        # Calculate mid price
        mid_price = (best_bid + best_ask) / 2
        
        # Keep track of position for limit checking
        current_position = position
        
        # PART 1: MAIN TRADING STRATEGY
        threshold = 1

        # Sort sell orders by price in ascending order (best prices first)
        for price, volume in sorted(order_depth.sell_orders.items()):
            # If price is below our threshold (mid - 1), we want to buy
            if price <= (mid_price - threshold):
                # Check position limit for buying
                buy_limit = position_limit - current_position
                
                if buy_limit > 0:  # Only if we can buy
                    # Buy either the available volume or our limit, whichever is smaller
                    # Note: volume is negative in sell_orders, so we negate it
                    buy_volume = min(-volume, buy_limit)
                    
                    if buy_volume > 0:
                        orders.append(Order("STARFRUIT", price, buy_volume))
                        current_position += buy_volume  # Update position tracker
        

        # Sort buy orders by price in descending order (best prices first)
        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            # If price is above our threshold (mid + 1), we want to sell
            if price >= (mid_price + threshold):
                # Check position limit for selling
                sell_limit = position_limit + current_position
                
                if sell_limit > 0:  # Only if we can sell
                    # Sell either the available volume or our limit, whichever is smaller
                    sell_volume = min(volume, sell_limit)
                    
                    if sell_volume > 0:
                        orders.append(Order("STARFRUIT", price, -sell_volume))
                        current_position -= sell_volume  # Update position tracker

        # Part 2: Try to clear positions additionally
        for price, volume in sorted(order_depth.sell_orders.items()):
            if price > (mid_price - threshold) and price <= mid_price:
                buy_limit = position_limit - current_position
                if buy_limit > 0:
                    buy_volume = min(-volume, buy_limit)
                    if buy_volume > 0:
                        orders.append(Order("STARFRUIT", price, buy_volume))
                        current_position += buy_volume

        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            if price < (mid_price + threshold) and price >= mid_price:
                sell_limit = position_limit + current_position
                if sell_limit > 0:
                    sell_volume = min(volume, sell_limit)
                    if sell_volume > 0:
                        orders.append(Order("STARFRUIT", price, -sell_volume))
                        current_position -= sell_volume
        
        # PART 3: PLACE RESTING ORDERS AT MID PRICE ± 3
        # Calculate resting order prices
        wide_threshold = 3
        rest_buy_price = int(mid_price - wide_threshold)  # Round down for buy
        rest_sell_price = int(mid_price + wide_threshold)
        if mid_price % 1 != 0:  # If mid price has decimal part, round up for sell
            rest_sell_price += 1
        
        # Calculate maximum volumes we can trade without breaching position limits
        max_buy_volume = position_limit - max(position, current_position)
        max_sell_volume = position_limit + min(position, current_position)
        
        # Place resting buy order
        if max_buy_volume > 0:
            orders.append(Order("STARFRUIT", rest_buy_price, max_buy_volume))
        
        # Place resting sell order
        if max_sell_volume > 0:
            orders.append(Order("STARFRUIT", rest_sell_price, -max_sell_volume))
        
        result["STARFRUIT"] = orders
        
        logger.flush(state, result, 0, "")
        return result, 0, ""

def Trader():
    """Factory function that returns a trader instance"""
    return StarfruitTrader()