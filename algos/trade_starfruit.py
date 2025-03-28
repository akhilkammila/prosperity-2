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
        # Initialize with empty values
        self.last_valid_best_bid = None
        self.last_valid_best_ask = None
        
    def parse_trader_data(self, trader_data: str) -> dict:
        """Parse the trader data string into a dictionary"""
        if not trader_data:
            return {"best_bid": None, "best_ask": None}
        
        try:
            return json.loads(trader_data)
        except json.JSONDecodeError:
            return {"best_bid": None, "best_ask": None}
    
    def serialize_trader_data(self, data: dict) -> str:
        """Serialize the trader data dictionary to a string"""
        return json.dumps(data)
    
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
        - Look at the orderbook to find favorable prices
        - Trade at those specific prices with corresponding volumes
        - Try to clear position at mid price if possible
        """
        result = {}
        
        # Only trade STARFRUIT
        if "STARFRUIT" not in state.order_depths:
            return result, 0, ""

        # Get current position and order depth for STARFRUIT
        position = state.position.get("STARFRUIT", 0)
        order_depth = state.order_depths["STARFRUIT"]
        
        # Parse trader data to get previous best prices
        trader_data = self.parse_trader_data(state.traderData)
        self.last_valid_best_bid = trader_data.get("best_bid")
        self.last_valid_best_ask = trader_data.get("best_ask")
        
        # Find best bid and ask with volume > 15
        best_bid, best_ask = self.find_best_prices_with_volume(order_depth)
        
        # Update last valid prices if we found valid ones
        if best_bid is not None:
            self.last_valid_best_bid = best_bid
        if best_ask is not None:
            self.last_valid_best_ask = best_ask
            
        # Create orders list
        orders: list[Order] = []
        
        # Position limit
        position_limit = 20
        
        # Only proceed if we have both a valid bid and ask
        if self.last_valid_best_bid is not None and self.last_valid_best_ask is not None:
            # Calculate mid price
            mid_price = (self.last_valid_best_bid + self.last_valid_best_ask) / 2
            mid_price_int = int(mid_price + 0.5)  # Round to nearest integer
            
            # Keep track of position for limit checking
            current_position = position
            
            # PART 1: MAIN TRADING STRATEGY
            # Process sell orders - if price < mid-1, we want to buy
            if len(order_depth.sell_orders) > 0:
                # Sort sell orders by price in ascending order (best prices first)
                for price, volume in sorted(order_depth.sell_orders.items()):
                    # If price is below our threshold (mid - 1), we want to buy
                    if price <= (mid_price - 1):
                        # Check position limit for buying
                        buy_limit = position_limit - current_position
                        
                        if buy_limit > 0:  # Only if we can buy
                            # Buy either the available volume or our limit, whichever is smaller
                            # Note: volume is negative in sell_orders, so we negate it
                            buy_volume = min(-volume, buy_limit)
                            
                            if buy_volume > 0:
                                orders.append(Order("STARFRUIT", price, buy_volume))
                                current_position += buy_volume  # Update position tracker
            
            # Process buy orders - if price > mid+1, we want to sell
            if len(order_depth.buy_orders) > 0:
                # Sort buy orders by price in descending order (best prices first)
                for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    # If price is above our threshold (mid + 1), we want to sell
                    if price >= (mid_price + 1):
                        # Check position limit for selling
                        sell_limit = position_limit + current_position
                        
                        if sell_limit > 0:  # Only if we can sell
                            # Sell either the available volume or our limit, whichever is smaller
                            sell_volume = min(volume, sell_limit)
                            
                            if sell_volume > 0:
                                orders.append(Order("STARFRUIT", price, -sell_volume))
                                current_position -= sell_volume  # Update position tracker
            
            # PART 2: POSITION CLEARING AT MID PRICE (THRESHOLD = 0)
            # If we're long, try to sell at mid price
            if current_position > 0:
                # Look for buy orders close to mid price
                for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    # Find orders close to mid price (can be slightly below or equal to)
                    if price >= mid_price_int - 0.5 and price <= mid_price_int + 0.5:
                        # Only sell up to the amount we are long by
                        clear_volume = min(volume, current_position)
                        if clear_volume > 0:
                            orders.append(Order("STARFRUIT", price, -clear_volume))
                            current_position -= clear_volume
                            if current_position <= 0:
                                break  # Stop if we've cleared our position
            
            # If we're short, try to buy at mid price
            elif current_position < 0:
                # Look for sell orders close to mid price
                for price, volume in sorted(order_depth.sell_orders.items()):
                    # Find orders close to mid price (can be slightly above or equal to)
                    if price >= mid_price_int - 0.5 and price <= mid_price_int + 0.5:
                        # Only buy up to the amount we are short by
                        clear_volume = min(-volume, abs(current_position))
                        if clear_volume > 0:
                            orders.append(Order("STARFRUIT", price, clear_volume))
                            current_position += clear_volume
                            if current_position >= 0:
                                break  # Stop if we've cleared our position
        
        result["STARFRUIT"] = orders
        
        # Save current best prices for next iteration
        new_trader_data = {
            "best_bid": self.last_valid_best_bid,
            "best_ask": self.last_valid_best_ask
        }
        
        trader_data_str = self.serialize_trader_data(new_trader_data)
        logger.flush(state, result, 0, trader_data_str)
        return result, 0, trader_data_str

def Trader():
    """Factory function that returns a trader instance"""
    return StarfruitTrader() 