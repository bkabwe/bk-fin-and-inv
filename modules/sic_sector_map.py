"""Map a Polygon/SEC SIC code onto the standard sector names the app keys on.

Polygon's ticker overview carries only a Standard Industrial Classification
(SIC) code and description (e.g. ``3571`` / "ELECTRONIC COMPUTERS"); it has no
sector field. Every sector-aware feature in the app -- the sector P/E benchmark
(modules.fundamental_analysis), sector momentum (modules.scoring_engine), the
SPDR sector-ETF relative-strength features (modules.sector_returns) and the
long-term sector trend -- keys on the eleven standard names below. Handing
them the raw SIC description ("SERVICES-PREPACKAGED SOFTWARE") matched none of
them, so each of those features silently did nothing.

SIC predates GICS and does not line up with it one-to-one, so this is a
best-effort crosswalk at 2-4 digit granularity. The first matching range wins,
so specific carve-outs are listed before the broad ranges that contain them.
A few SIC catch-alls land on the closest sector rather than an exact GICS one
(7370 "data processing" -> Technology and 7389 "business services NEC" ->
Industrials, so internet platforms and payment networks can sit one sector
away from their GICS home). A code with no rule returns ``None``, which
downstream code already treats as "sector unknown".
"""

from __future__ import annotations

from typing import Any

TECHNOLOGY = "Technology"
HEALTHCARE = "Healthcare"
FINANCIALS = "Financials"
ENERGY = "Energy"
CONSUMER_STAPLES = "Consumer Staples"
CONSUMER_DISCRETIONARY = "Consumer Discretionary"
INDUSTRIALS = "Industrials"
MATERIALS = "Materials"
UTILITIES = "Utilities"
REAL_ESTATE = "Real Estate"
COMMUNICATION_SERVICES = "Communication Services"

# (first SIC code, last SIC code, sector), both ends inclusive, first match wins.
_SIC_SECTOR_RULES: tuple[tuple[int, int, str], ...] = (
    # Agriculture, forestry, fishing (01xx-09xx).
    (800, 899, MATERIALS),  # forestry
    (100, 999, CONSUMER_STAPLES),
    # Mining (10xx-14xx).
    (1000, 1099, MATERIALS),  # metal mining
    (1200, 1399, ENERGY),  # coal, crude petroleum & natural gas, drilling, field services
    (1400, 1499, MATERIALS),  # stone, sand, phosphate and other nonmetallic minerals
    # Construction (15xx-17xx).
    (1520, 1531, CONSUMER_DISCRETIONARY),  # residential builders
    (1500, 1799, INDUSTRIALS),
    # Food, tobacco, textiles, apparel (20xx-23xx).
    (2000, 2199, CONSUMER_STAPLES),
    (2200, 2399, CONSUMER_DISCRETIONARY),
    # Wood, furniture, paper (24xx-26xx).
    (2451, 2452, CONSUMER_DISCRETIONARY),  # mobile homes, prefabricated wood buildings
    (2400, 2499, MATERIALS),
    (2500, 2599, CONSUMER_DISCRETIONARY),
    (2676, 2676, CONSUMER_STAPLES),  # sanitary paper products (tissue, diapers)
    (2600, 2699, MATERIALS),
    # Publishing and printing (27xx).
    (2700, 2749, COMMUNICATION_SERVICES),
    (2750, 2799, INDUSTRIALS),  # commercial printing, business forms
    # Chemicals (28xx).
    (2830, 2836, HEALTHCARE),  # pharmaceuticals, biologicals, diagnostic substances
    (2840, 2849, CONSUMER_STAPLES),  # soap, cleaning products, cosmetics
    (2800, 2899, MATERIALS),
    # Petroleum refining (29xx).
    (2900, 2999, ENERGY),
    # Rubber/plastics, leather, stone/glass, primary metals (30xx-33xx).
    (3011, 3011, CONSUMER_DISCRETIONARY),  # tires
    (3021, 3021, CONSUMER_DISCRETIONARY),  # rubber & plastics footwear
    (3000, 3099, MATERIALS),
    (3100, 3199, CONSUMER_DISCRETIONARY),
    (3200, 3399, MATERIALS),
    # Fabricated metals, machinery, computers (34xx-35xx).
    (3411, 3412, MATERIALS),  # metal cans and shipping containers
    (3400, 3499, INDUSTRIALS),
    (3533, 3533, ENERGY),  # oil & gas field machinery
    (3559, 3559, TECHNOLOGY),  # special industry machinery (semiconductor equipment)
    (3570, 3579, TECHNOLOGY),  # computers and peripherals
    (3500, 3599, INDUSTRIALS),
    # Electrical and electronic equipment (36xx).
    (3630, 3639, CONSUMER_DISCRETIONARY),  # household appliances
    (3651, 3651, CONSUMER_DISCRETIONARY),  # household audio & video equipment
    (3652, 3652, COMMUNICATION_SERVICES),  # prerecorded audio media
    (3661, 3679, TECHNOLOGY),  # communications equipment, semiconductors, electronic components
    (3695, 3695, TECHNOLOGY),  # magnetic & optical recording media
    (3600, 3699, INDUSTRIALS),
    # Transportation equipment (37xx).
    (3710, 3711, CONSUMER_DISCRETIONARY),  # motor vehicles
    (3714, 3714, CONSUMER_DISCRETIONARY),  # motor vehicle parts
    (3716, 3716, CONSUMER_DISCRETIONARY),  # motor homes
    (3732, 3732, CONSUMER_DISCRETIONARY),  # boat building
    (3751, 3751, CONSUMER_DISCRETIONARY),  # motorcycles, bicycles
    (3792, 3792, CONSUMER_DISCRETIONARY),  # travel trailers, campers
    (3799, 3799, CONSUMER_DISCRETIONARY),  # recreational vehicles NEC
    (3700, 3799, INDUSTRIALS),  # aerospace & defense, rail, trucks and the rest
    # Instruments, medical and optical goods (38xx).
    (3810, 3819, INDUSTRIALS),  # navigation, defense electronics, scientific instruments
    (3820, 3821, HEALTHCARE),  # laboratory apparatus
    (3822, 3822, INDUSTRIALS),  # environmental controls
    (3826, 3826, HEALTHCARE),  # laboratory analytical instruments
    (3840, 3859, HEALTHCARE),  # surgical, dental, electromedical, ophthalmic
    (3870, 3879, CONSUMER_DISCRETIONARY),  # watches and clocks
    (3800, 3899, TECHNOLOGY),  # remaining measuring/test instruments, photographic equipment
    # Miscellaneous manufacturing: jewelry, toys, sporting goods (39xx).
    (3900, 3999, CONSUMER_DISCRETIONARY),
    # Transportation, communications, utilities (40xx-49xx).
    (4000, 4599, INDUSTRIALS),  # rail, trucking, shipping, airlines
    (4600, 4699, ENERGY),  # pipelines
    (4700, 4729, CONSUMER_DISCRETIONARY),  # travel agencies, tour operators
    (4730, 4799, INDUSTRIALS),  # freight arrangement and other transportation services
    (4800, 4899, COMMUNICATION_SERVICES),
    (4922, 4922, ENERGY),  # natural gas transmission
    (4950, 4959, INDUSTRIALS),  # sanitary services (waste management)
    (4900, 4999, UTILITIES),
    # Wholesale trade (50xx-51xx).
    (5010, 5029, CONSUMER_DISCRETIONARY),  # auto parts, furniture and home furnishings
    (5045, 5045, TECHNOLOGY),  # computers and software
    (5047, 5047, HEALTHCARE),  # medical, dental & hospital equipment
    (5050, 5059, MATERIALS),  # metals & minerals
    (5064, 5064, CONSUMER_DISCRETIONARY),  # electrical appliances, TV & radio
    (5065, 5065, TECHNOLOGY),  # electronic parts
    (5090, 5099, CONSUMER_DISCRETIONARY),  # misc durable goods
    (5000, 5099, INDUSTRIALS),
    (5110, 5119, INDUSTRIALS),  # paper & paper products
    (5122, 5122, HEALTHCARE),  # drugs & druggists' sundries
    (5130, 5139, CONSUMER_DISCRETIONARY),  # apparel, piece goods, footwear
    (5160, 5169, MATERIALS),  # chemicals
    (5171, 5172, ENERGY),  # petroleum bulk stations and products
    (5100, 5199, CONSUMER_STAPLES),  # groceries, farm products, beer & wine
    # Retail trade (52xx-59xx).
    (5330, 5339, CONSUMER_STAPLES),  # variety stores
    (5399, 5399, CONSUMER_STAPLES),  # warehouse clubs and misc general merchandise
    (5400, 5499, CONSUMER_STAPLES),  # food stores
    (5912, 5912, CONSUMER_STAPLES),  # drug stores
    (5200, 5999, CONSUMER_DISCRETIONARY),
    # Finance, insurance, real estate (60xx-67xx).
    (6324, 6324, HEALTHCARE),  # hospital & medical service plans (managed care)
    (6500, 6599, REAL_ESTATE),
    (6792, 6792, ENERGY),  # oil royalty traders
    (6798, 6798, REAL_ESTATE),  # REITs
    (6000, 6799, FINANCIALS),
    # Services (70xx-89xx).
    (7000, 7099, CONSUMER_DISCRETIONARY),  # hotels and lodging
    (7200, 7299, CONSUMER_DISCRETIONARY),  # personal services
    (7310, 7319, COMMUNICATION_SERVICES),  # advertising
    (7370, 7379, TECHNOLOGY),  # software, IT services, data processing
    (7300, 7399, INDUSTRIALS),  # other business services (7389 "NEC" lands here)
    (7510, 7519, INDUSTRIALS),  # auto & truck rental
    (7500, 7599, CONSUMER_DISCRETIONARY),
    (7600, 7699, INDUSTRIALS),
    (7800, 7899, COMMUNICATION_SERVICES),  # motion pictures, theaters, video
    (7900, 7999, CONSUMER_DISCRETIONARY),  # amusement and recreation
    (8000, 8099, HEALTHCARE),
    (8100, 8199, INDUSTRIALS),
    (8200, 8399, CONSUMER_DISCRETIONARY),  # education and child care
    (8731, 8731, HEALTHCARE),  # commercial physical & biological research
    (8700, 8999, INDUSTRIALS),
)


def _parse_sic_code(value: Any) -> int | None:
    """Parse a SIC code given as ``"3571"``, ``3571`` or ``"0100"``; ``None``
    for anything that is not a whole number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def sector_from_sic(sic_code: Any) -> str | None:
    """Return the standard sector name for a SIC code, or ``None`` when the
    code is missing, malformed or has no rule (callers treat that as an
    unknown sector)."""
    code = _parse_sic_code(sic_code)
    if code is None:
        return None
    for low, high, sector in _SIC_SECTOR_RULES:
        if low <= code <= high:
            return sector
    return None
