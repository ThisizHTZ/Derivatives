"""金融衍生品定价工具包。"""

from derivatives.pricing.bsm import black_scholes_merton
from derivatives.pricing.binary import (
    asset_or_nothing_call,
    asset_or_nothing_put,
    binary_option_price,
    cash_or_nothing_call,
    cash_or_nothing_put,
    supershare_option_price,
)
from derivatives.pricing.barrier import (
    barrier_option_price,
    binary_barrier_option,
    double_barrier_call,
    double_barrier_put,
)
from derivatives.pricing.knock_in import down_and_in_call, up_and_in_call
from derivatives.pricing.structured import (
    airbag_option_price,
    range_accrual_price,
    shark_fin_option,
    up_and_out_call,
)
from derivatives.pricing.exotic import (
    StructuredNoteResult,
    arithmetic_asian_option_price,
    basket_option_price,
    capital_protected_note_price,
    fixed_strike_lookback_option_price,
    phoenix_autocall_note_price,
    snowball_note_price,
)
from derivatives.models.binomial_tree import european_call_tree, european_put_tree, american_option_tree
from derivatives.models.monte_carlo import european_call_mc, geo_brownian_motion
from derivatives.models.brownian import standard_brownian_motion
from derivatives.analytics.greeks import compute_greeks
from derivatives.analytics.basis_rate import compute_basis_rates

__all__ = [
    "black_scholes_merton",
    "binary_option_price",
    "cash_or_nothing_call",
    "cash_or_nothing_put",
    "asset_or_nothing_call",
    "asset_or_nothing_put",
    "supershare_option_price",
    "double_barrier_call",
    "double_barrier_put",
    "barrier_option_price",
    "binary_barrier_option",
    "down_and_in_call",
    "up_and_in_call",
    "shark_fin_option",
    "up_and_out_call",
    "airbag_option_price",
    "range_accrual_price",
    "StructuredNoteResult",
    "capital_protected_note_price",
    "arithmetic_asian_option_price",
    "fixed_strike_lookback_option_price",
    "basket_option_price",
    "snowball_note_price",
    "phoenix_autocall_note_price",
    "european_call_tree",
    "european_put_tree",
    "american_option_tree",
    "european_call_mc",
    "geo_brownian_motion",
    "standard_brownian_motion",
    "compute_greeks",
    "compute_basis_rates",
]

__version__ = "0.3.0"
