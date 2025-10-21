from __future__ import annotations
from typing import Dict, List


# -----------------------------------------------------------------------------
# Canonical name → categories (lowercase). Keep names human-friendly & specific.
# LLM mode may override these; we still keep as fallback + normalizer source.
# -----------------------------------------------------------------------------
NAME_TO_CATS: Dict[str, List[str]] = {
    # Liquids & containers
    "water": ["liquid", "water"],
    "cup": ["container"],
    "cup of water": ["liquid", "water", "container"],
    "mug": ["container"],
    "bottle": ["container"],
    "bottle of water": ["liquid", "water", "container"],
    "coffee": ["liquid"],
    "tea": ["liquid"],
    "oil": ["liquid", "flammable"],


    # Electronics
    "laptop": ["electronics"],
    "phone": ["electronics"],
    "tablet": ["electronics"],
    "pc": ["electronics"],
    "desktop": ["electronics"],
    "keyboard": ["electronics"],
    "monitor": ["fragile", "electronics"],
    "camera": ["electronics", "fragile"],
    "speaker": ["electronics"],


    # Heat sources & flammables
    "candle": ["heat", "flame"],
    "stove": ["heat"],
    "hot plate": ["heat"],
    "oven": ["heat"],
    "soldering iron": ["heat"],
    "lighter": ["heat", "flame"],
    "alcohol": ["flammable", "liquid"],
    "paper": ["flammable"],
    "cardboard": ["flammable"],
    "cloth": ["flammable"],
    "plastic bag": ["flammable"],
    "wood": ["flammable"],


    # Sharp & human
    "knife": ["sharp"],
    "scissors": ["sharp"],
    "razor": ["sharp"],
    "needle": ["sharp"],
    "saw": ["sharp"],
    "hand": ["human", "skin"],
    "arm": ["human", "skin"],
    "finger": ["human", "skin"],
    "person": ["human"],


    # Fragile (non-electronics)
    "glass": ["fragile"],
    "vase": ["fragile"],
    "plate": ["fragile"],
    "cup (glass)": ["fragile", "container"],


    # Toxic & food
    "bleach": ["toxic", "liquid"],
    "ammonia": ["toxic", "liquid"],
    "pesticide": ["toxic"],
    "detergent": ["toxic"],
    "apple": ["food"],
    "sandwich": ["food"],
    "bread": ["food"],


    # Magnet & cards
    "magnet": ["magnet"],
    "credit card": ["card"],
    "hotel key card": ["card"],
    "id card": ["card"],


    # Voltage & power
    "power strip": ["voltage", "electronics"],
    "outlet": ["voltage"],
    "extension cord": ["voltage"],
    "charger": ["voltage", "electronics"],
    "battery": ["voltage"],
}

CATEGORY_DEFAULTS = {
    "fragile": {"fragile": True},
}