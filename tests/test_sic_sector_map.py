from __future__ import annotations

import unittest

from modules import sic_sector_map
from modules.fundamental_analysis import SECTOR_BENCHMARK_PE, normalize_sector_name
from modules.sector_returns import SECTOR_ETF_MAP
from modules.sic_sector_map import sector_from_sic


class SectorFromSicTests(unittest.TestCase):
    def test_maps_representative_sic_codes_to_the_standard_sectors(self):
        cases = {
            "3571": "Technology",  # electronic computers
            "3674": "Technology",  # semiconductors
            "7372": "Technology",  # prepackaged software
            "2834": "Healthcare",  # pharmaceutical preparations
            "3841": "Healthcare",  # surgical & medical instruments
            "8062": "Healthcare",  # hospitals
            "6021": "Financials",  # national commercial banks
            "6331": "Financials",  # fire, marine & casualty insurance
            "6211": "Financials",  # security brokers & dealers
            "6798": "Real Estate",  # REITs
            "1311": "Energy",  # crude petroleum & natural gas
            "2911": "Energy",  # petroleum refining
            "4911": "Utilities",  # electric services
            "4941": "Utilities",  # water supply
            "4813": "Communication Services",  # telephone communications
            "4841": "Communication Services",  # cable & other pay television
            "7841": "Communication Services",  # video tape rental (streaming)
            "2080": "Consumer Staples",  # beverages
            "5411": "Consumer Staples",  # grocery stores
            "5331": "Consumer Staples",  # variety stores
            "5812": "Consumer Discretionary",  # eating places
            "5961": "Consumer Discretionary",  # catalog & mail-order houses
            "3711": "Consumer Discretionary",  # motor vehicles
            "7011": "Consumer Discretionary",  # hotels & motels
            "3721": "Industrials",  # aircraft
            "3531": "Industrials",  # construction machinery
            "4512": "Industrials",  # scheduled air transportation
            "7389": "Industrials",  # business services NEC
            "1040": "Materials",  # gold & silver ores
            "2810": "Materials",  # industrial inorganic chemicals
            "3312": "Materials",  # steel works
        }
        for code, expected in cases.items():
            with self.subTest(sic_code=code):
                self.assertEqual(sector_from_sic(code), expected)

    def test_specific_carve_outs_beat_the_broad_range_that_contains_them(self):
        pairs = [
            # (carve-out code, sector), (neighbouring code in the broad range, sector)
            (("1531", "Consumer Discretionary"), ("1600", "Industrials")),  # residential builders
            (("3559", "Technology"), ("3560", "Industrials")),  # semiconductor equipment
            (("3533", "Energy"), ("3530", "Industrials")),  # oil & gas field machinery
            (("3826", "Healthcare"), ("3825", "Technology")),  # lab instruments vs. electrical test gear
            (("2833", "Healthcare"), ("2820", "Materials")),  # drugs vs. plastics & resins
            (("2844", "Consumer Staples"), ("2870", "Materials")),  # cosmetics vs. agricultural chemicals
            (("2676", "Consumer Staples"), ("2650", "Materials")),  # tissue vs. paperboard containers
            (("0800", "Materials"), ("0100", "Consumer Staples")),  # forestry vs. crops
            (("4922", "Energy"), ("4911", "Utilities")),  # gas pipelines vs. electric utilities
            (("4953", "Industrials"), ("4941", "Utilities")),  # waste management vs. water utilities
            (("6324", "Healthcare"), ("6331", "Financials")),  # managed care vs. insurers
            (("5912", "Consumer Staples"), ("5961", "Consumer Discretionary")),  # drug stores vs. other retail
            (("5122", "Healthcare"), ("5141", "Consumer Staples")),  # drug wholesalers vs. food wholesalers
            (("3011", "Consumer Discretionary"), ("3089", "Materials")),  # tires vs. plastic products
        ]
        for (carve_code, carve_sector), (broad_code, broad_sector) in pairs:
            with self.subTest(carve_out=carve_code, broad=broad_code):
                self.assertEqual(sector_from_sic(carve_code), carve_sector)
                self.assertEqual(sector_from_sic(broad_code), broad_sector)

    def test_accepts_int_float_and_zero_padded_codes(self):
        self.assertEqual(sector_from_sic(3571), "Technology")
        self.assertEqual(sector_from_sic(3571.0), "Technology")
        self.assertEqual(sector_from_sic(" 3571 "), "Technology")
        self.assertEqual(sector_from_sic("0100"), "Consumer Staples")
        self.assertEqual(sector_from_sic(100), "Consumer Staples")

    def test_returns_none_for_missing_malformed_or_unmapped_codes(self):
        for value in (None, "", "   ", "abc", "35 71", "3571.5", 3571.5, float("nan"), float("inf"), True, [], {}):
            with self.subTest(value=value):
                self.assertIsNone(sector_from_sic(value))
        # Public-administration / non-operating-establishment codes and codes
        # outside the SIC range have no sector.
        for code in ("9995", "9721", "0", "99999", "-3571"):
            with self.subTest(code=code):
                self.assertIsNone(sector_from_sic(code))


class SicSectorRuleTableTests(unittest.TestCase):
    def test_rules_are_well_formed(self):
        for low, high, sector in sic_sector_map._SIC_SECTOR_RULES:
            with self.subTest(low=low, high=high):
                self.assertLessEqual(0, low)
                self.assertLessEqual(low, high)
                self.assertLessEqual(high, 9999)
                self.assertIsInstance(sector, str)

    def test_every_sector_is_one_the_sector_features_recognize(self):
        standard = set(SECTOR_BENCHMARK_PE)
        self.assertEqual(standard, set(SECTOR_ETF_MAP))
        used = {sector for _, _, sector in sic_sector_map._SIC_SECTOR_RULES}
        # Every label survives normalization unchanged and has both a P/E
        # benchmark and a sector ETF, so the features that key on it all work;
        # and every standard sector is reachable from some SIC code.
        self.assertEqual(used, standard)
        for sector in used:
            self.assertEqual(normalize_sector_name(sector), sector)

    def test_no_rule_is_fully_shadowed_by_earlier_rules(self):
        # Rules are first-match-wins, so a carve-out listed after the broad
        # range that contains it would never fire. Every rule must win for at
        # least one code.
        winners: set[int] = set()
        rules = sic_sector_map._SIC_SECTOR_RULES
        for code in range(0, 10000):
            for index, (low, high, _) in enumerate(rules):
                if low <= code <= high:
                    winners.add(index)
                    break
        shadowed = [rules[i] for i in range(len(rules)) if i not in winners]
        self.assertEqual(shadowed, [])


if __name__ == "__main__":
    unittest.main()
