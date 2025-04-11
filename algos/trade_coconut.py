import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, Dict, List
import math
from scipy.stats import norm

# Import option pricing function
def call_option_value(current_price, strike=10000, ticks=250*10000):
    sigma = math.sqrt(ticks)  # Standard deviation after 250 ticks
    d = (current_price - strike) / sigma
    option_price = (current_price - strike) * norm.cdf(d) + sigma * norm.pdf(d)
    return option_price

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

class CoconutTrader:
    def __init__(self):
        pass
    
    def calculate_mid_price(self, order_depth: OrderDepth):
        """Calculate mid price from best bid and ask prices"""
        best_bid = None
        best_ask = None
        
        # Find best bid (highest buy price)
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
        
        # Find best ask (lowest sell price)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
        
        # If we have both bid and ask, calculate mid price
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2
        # If we only have bid
        elif best_bid is not None:
            return best_bid
        # If we only have ask
        elif best_ask is not None:
            return best_ask
        # If we have neither
        else:
            return None

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Trading strategy for COCONUT and COCONUT_COUPON:
        - Calculate fair value of COCONUT_COUPON using option pricing model
        - Trade COCONUT_COUPON based on price difference from fair value
        - Manage COCONUT position to be 0.53 * COCONUT_COUPON position
        """
        result = {}
        orders: Dict[Symbol, List[Order]] = {}
        orders["COCONUT"] = []
        orders["COCONUT_COUPON"] = []
        
        # Get current positions
        coconut_position = state.position.get("COCONUT", 0)
        coupon_position = state.position.get("COCONUT_COUPON", 0)
        
        # Position limits
        coconut_limit = 300
        coupon_limit = 600
        
        # Check if both symbols are available in order depths
        if "COCONUT" not in state.order_depths or "COCONUT_COUPON" not in state.order_depths:
            logger.flush(state, orders, 0, "")
            return orders, 0, ""
        
        # Get mid prices
        coconut_mid = self.calculate_mid_price(state.order_depths["COCONUT"])
        coupon_mid = self.calculate_mid_price(state.order_depths["COCONUT_COUPON"])
        
        if coconut_mid is None or coupon_mid is None:
            logger.flush(state, orders, 0, "")
            return orders, 0, ""
        
        # Calculate fair value of COCONUT_COUPON using option pricing
        fair_coupon_value = call_option_value(coconut_mid)
        
        # Log mid prices and fair value
        logger.print(f"COCONUT mid price: {coconut_mid}")
        logger.print(f"COCONUT_COUPON mid price: {coupon_mid}")
        logger.print(f"COCONUT_COUPON fair value: {fair_coupon_value}")
        logger.print(f"Price difference: {coupon_mid - fair_coupon_value}")
        
        # Get order depths
        coconut_depth = state.order_depths["COCONUT"]
        coupon_depth = state.order_depths["COCONUT_COUPON"]
        
        # Expected new position for COCONUT_COUPON (start with current position)
        expected_coupon_position = coupon_position
        
        # Trading logic for COCONUT_COUPON
        threshold = 10
        exit_threshold = 1
        
        # Check if we need to exit positions when price difference is small
        if abs(coupon_mid - fair_coupon_value) < exit_threshold:
            # If we have a long position in COCONUT_COUPON, exit
            if coupon_position > 0 and coupon_depth.buy_orders:
                best_bid_price = max(coupon_depth.buy_orders.keys())
                best_bid_volume = coupon_depth.buy_orders[best_bid_price]
                
                # Calculate exit volume (limited by current position and available market volume)
                exit_volume = min(coupon_position, best_bid_volume)
                
                if exit_volume > 0:
                    # Sell COCONUT_COUPON to exit long position
                    orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_bid_price, -exit_volume))
                    expected_coupon_position -= exit_volume
                    logger.print(f"Exiting long position: Selling {exit_volume} COCONUT_COUPON at {best_bid_price}")
            
            # If we have a short position in COCONUT_COUPON, exit
            elif coupon_position < 0 and coupon_depth.sell_orders:
                best_ask_price = min(coupon_depth.sell_orders.keys())
                best_ask_volume = -coupon_depth.sell_orders[best_ask_price]  # Negate since volumes are negative
                
                # Calculate exit volume (limited by current position and available market volume)
                exit_volume = min(-coupon_position, best_ask_volume)
                
                if exit_volume > 0:
                    # Buy COCONUT_COUPON to exit short position
                    orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_ask_price, exit_volume))
                    expected_coupon_position += exit_volume
                    logger.print(f"Exiting short position: Buying {exit_volume} COCONUT_COUPON at {best_ask_price}")
        
        # Regular trading logic when price difference is significant
        elif coupon_mid - fair_coupon_value > threshold:  # COCONUT_COUPON is overpriced -> SELL
            # Sell COCONUT_COUPON if overpriced
            if coupon_depth.buy_orders and coupon_position > -coupon_limit:
                best_bid_price = max(coupon_depth.buy_orders.keys())
                best_bid_volume = coupon_depth.buy_orders[best_bid_price]
                
                # Calculate how much we can sell without exceeding position limit
                max_sell_volume = min(best_bid_volume, coupon_limit + coupon_position)
                
                if max_sell_volume > 0:
                    # Sell COCONUT_COUPON
                    orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_bid_price, -max_sell_volume))
                    # Update expected position
                    expected_coupon_position -= max_sell_volume
        
        elif fair_coupon_value - coupon_mid > threshold:  # COCONUT_COUPON is underpriced -> BUY
            # Buy COCONUT_COUPON if underpriced
            if coupon_depth.sell_orders and coupon_position < coupon_limit:
                best_ask_price = min(coupon_depth.sell_orders.keys())
                best_ask_volume = -coupon_depth.sell_orders[best_ask_price]  # Negate since volumes are negative
                
                # Calculate how much we can buy without exceeding position limit
                max_buy_volume = min(best_ask_volume, coupon_limit - coupon_position)
                
                if max_buy_volume > 0:
                    # Buy COCONUT_COUPON
                    orders["COCONUT_COUPON"].append(Order("COCONUT_COUPON", best_ask_price, max_buy_volume))
                    # Update expected position
                    expected_coupon_position += max_buy_volume
        
        # Calculate desired COCONUT position based on expected COCONUT_COUPON position
        desired_coconut_position = round(expected_coupon_position * (-0.53))
        logger.print(f"Current COCONUT position: {coconut_position}")
        logger.print(f"Expected COCONUT_COUPON position: {expected_coupon_position}")
        logger.print(f"Desired COCONUT position: {desired_coconut_position}")
        
        # Calculate needed COCONUT position change
        coconut_position_change = desired_coconut_position - coconut_position
        
        # # COCONUT trading logic based on desired position
        # if coconut_position_change > 0:  # Need to BUY COCONUT
        #     if coconut_depth.sell_orders and coconut_position < coconut_limit:
        #         best_ask_price = min(coconut_depth.sell_orders.keys())
        #         available_volume = -coconut_depth.sell_orders[best_ask_price]  # Negate since volumes are negative
                
        #         # Calculate how much we can buy (limited by available volume and position limit)
        #         buy_volume = min(available_volume, coconut_position_change, coconut_limit - coconut_position)
                
        #         if buy_volume > 0:
        #             orders["COCONUT"].append(Order("COCONUT", best_ask_price, buy_volume))
        #             logger.print(f"Buying {buy_volume} COCONUT at {best_ask_price}")
                
        # elif coconut_position_change < 0:  # Need to SELL COCONUT
        #     if coconut_depth.buy_orders and coconut_position > -coconut_limit:
        #         best_bid_price = max(coconut_depth.buy_orders.keys())
        #         available_volume = coconut_depth.buy_orders[best_bid_price]
                
        #         # Calculate how much we can sell (limited by available volume and position limit)
        #         sell_volume = min(available_volume, -coconut_position_change, coconut_limit + coconut_position)
                
        #         if sell_volume > 0:
        #             orders["COCONUT"].append(Order("COCONUT", best_bid_price, -sell_volume))
        #             logger.print(f"Selling {sell_volume} COCONUT at {best_bid_price}")
        
        # Return empty lists if no orders
        if not orders["COCONUT"]:
            del orders["COCONUT"]
        if not orders["COCONUT_COUPON"]:
            del orders["COCONUT_COUPON"]
        
        logger.flush(state, orders, 0, "")
        return orders, 0, ""

def Trader():
    """Factory function that returns a trader instance"""
    return CoconutTrader() 