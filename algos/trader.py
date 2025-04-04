import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

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

class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        """
        Trading strategy for AMETHYSTS:
        - Sell when price > 10,000
        - Buy when price < 10,000
        """
        result = {}
        
        # Only trade AMETHYSTS
        if "AMETHYSTS" not in state.order_depths:
            return result, 0, ""

        # Get current position and order depth for AMETHYSTS
        position = state.position.get("AMETHYSTS", 0)
        order_depth = state.order_depths["AMETHYSTS"]
        orders: list[Order] = []
        
        # Process buy orders - if price > 10000, we want to sell
        if len(order_depth.buy_orders) > 0:
            # Sort buy orders by price in descending order
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                if price > 10000:
                    # Check position limit for selling
                    # We can sell up to: -(POSITION_LIMIT + current_position)
                    sell_limit = 20 + position  # Position limit is 20 for AMETHYSTS
                    if sell_limit >= 0:  # Only if we can sell
                        # Sell either the available volume or our limit, whichever is smaller
                        sell_volume = min(volume, sell_limit)
                        if sell_volume > 0:
                            orders.append(Order("AMETHYSTS", price, -sell_volume))
                            position -= sell_volume  # Update our position tracker

        # Process sell orders - if price < 10000, we want to buy
        if len(order_depth.sell_orders) > 0:
            # Sort sell orders by price in ascending order
            for price, volume in sorted(order_depth.sell_orders.items()):
                if price < 10000:
                    # Check position limit for buying
                    # We can buy up to: POSITION_LIMIT - current_position
                    buy_limit = 20 - position  # Position limit is 20 for AMETHYSTS
                    if buy_limit > 0:  # Only if we can buy
                        # Buy either the available volume or our limit, whichever is smaller
                        # Note: volume is negative in sell_orders, so we negate it
                        buy_volume = min(-volume, buy_limit)
                        if buy_volume > 0:
                            orders.append(Order("AMETHYSTS", price, buy_volume))
                            position += buy_volume  # Update our position tracker

        result["AMETHYSTS"] = orders

        logger.flush(state, result, 0, "")
        return result, 0, ""