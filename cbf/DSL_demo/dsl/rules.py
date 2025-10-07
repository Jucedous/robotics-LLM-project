from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List
from .cbf import distance_barrier, above_barrier, closing_speed_barrier, barrier_to_risk

@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    match: Callable[[Dict], bool]
    barrier: Callable[[Dict], float]
    risk_scale: float = 0.1
    weight: float = 1.0

def has(cats: List[str], tag: str) -> bool:
    return tag in cats

RULES: List[Rule] = []

# 1) Liquid above Electronics
RULES.append(Rule(
    rule_id="R1",
    description="Liquid above electronics (risk of spill).",
    match=lambda ctx: has(ctx['catA'], 'liquid') and has(ctx['catB'], 'electronics') and ctx['rel'] == 'above',
    barrier=lambda ctx: above_barrier(ctx['posA'], ctx['posB'], z_min=0.2, r_align=0.25),
    risk_scale=0.1,
    weight=1.5,
))

# 2) Heat near Flammable
RULES.append(Rule(
    rule_id="R2",
    description="Heat source near flammable object (fire hazard).",
    match=lambda ctx: has(ctx['catA'], 'heat') and has(ctx['catB'], 'flammable') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.5),
    risk_scale=0.15,
    weight=1.3,
))

# 3) Sharp near Human skin
RULES.append(Rule(
    rule_id="R3",
    description="Sharp object near human/skin (cut hazard).",
    match=lambda ctx: has(ctx['catA'], 'sharp') and has(ctx['catB'], 'human') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.25),
    risk_scale=0.08,
    weight=1.2,
))

# 4) Heavy above Fragile
RULES.append(Rule(
    rule_id="R4",
    description="Heavy object above fragile item (crush hazard).",
    match=lambda ctx: ctx['objA'].mass and ctx['objA'].mass > 3.0 and has(ctx['catB'], 'fragile') and ctx['rel'] == 'above',
    barrier=lambda ctx: above_barrier(ctx['posA'], ctx['posB'], z_min=0.25, r_align=0.3),
    risk_scale=0.12,
    weight=1.1,
))

# 5) Toxic near Food
RULES.append(Rule(
    rule_id="R5",
    description="Toxic substance near food (contamination hazard).",
    match=lambda ctx: has(ctx['catA'], 'toxic') and has(ctx['catB'], 'food') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.4),
    risk_scale=0.12,
    weight=1.0,
))

# 6) Magnet near Card
RULES.append(Rule(
    rule_id="R6",
    description="Magnet near magnetic stripe card (data loss).",
    match=lambda ctx: has(ctx['catA'], 'magnet') and has(ctx['catB'], 'card') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.1),
    risk_scale=0.06,
    weight=0.8,
))

# 7) Voltage near Water
RULES.append(Rule(
    rule_id="R7",
    description="Voltage source near water (shock hazard).",
    match=lambda ctx: has(ctx['catA'], 'voltage') and has(ctx['catB'], 'water') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.3),
    risk_scale=0.12,
    weight=1.4,
))

# 8) Robot approaching Human too fast
RULES.append(Rule(
    rule_id="R8",
    description="Robot end-effector approaching human too fast (impact hazard).",
    match=lambda ctx: has(ctx['catA'], 'robot') and has(ctx['catB'], 'human') and ctx['rel'] == 'near' and (ctx['velA'] is not None),
    barrier=lambda ctx: closing_speed_barrier(ctx['posA'], ctx['velA'], ctx['posB'], v_max=0.15),
    risk_scale=0.1,
    weight=1.6,
))

# 9) Liquid near Electronics
RULES.append(Rule(
    rule_id="R9",
    description="Liquid close to electronics (splash/knock hazard).",
    match=lambda ctx: has(ctx['catA'], 'liquid') and has(ctx['catB'], 'electronics') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.35),
    risk_scale=0.15,
    weight=1.0,
))

# 10) Heat above Fragile/Electronics
RULES.append(Rule(
    rule_id="R10",
    description="Heat source above fragile/electronics (thermal hazard).",
    match=lambda ctx: has(ctx['catA'], 'heat') and (has(ctx['catB'], 'fragile') or has(ctx['catB'], 'electronics')) and ctx['rel'] == 'above',
    barrier=lambda ctx: above_barrier(ctx['posA'], ctx['posB'], z_min=0.15, r_align=0.2),
    risk_scale=0.1,
    weight=1.0,
))

# 11) Fragile near Hard surface
RULES.append(Rule(
    rule_id="R11",
    description="Fragile item near hard surface (breakage risk).",
    match=lambda ctx: has(ctx['catA'], 'fragile') and has(ctx['catB'], 'object') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.15),
    risk_scale=0.12,
    weight=1.0,
))

# 12) Toxic near Human
RULES.append(Rule(
    rule_id="R12",
    description="Toxic substance near human (poison exposure).",
    match=lambda ctx: has(ctx['catA'], 'toxic') and has(ctx['catB'], 'human') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.5),
    risk_scale=0.2,
    weight=1.5,
))

# 13) Heat near Electronics
RULES.append(Rule(
    rule_id="R13",
    description="Heat source near electronics (overheating hazard).",
    match=lambda ctx: has(ctx['catA'], 'heat') and has(ctx['catB'], 'electronics') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.4),
    risk_scale=0.15,
    weight=1.3,
))

# 14) Magnet near Electronics
RULES.append(Rule(
    rule_id="R14",
    description="Magnet near electronics (interference hazard).",
    match=lambda ctx: has(ctx['catA'], 'magnet') and has(ctx['catB'], 'electronics') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.2),
    risk_scale=0.1,
    weight=1.0,
))

# 15) Food near Unsanitary object
RULES.append(Rule(
    rule_id="R15",
    description="Food near toxic/sharp/unsanitary objects (contamination).",
    match=lambda ctx: has(ctx['catA'], 'food') and (has(ctx['catB'], 'toxic') or has(ctx['catB'], 'sharp')) and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.25),
    risk_scale=0.1,
    weight=1.0,
))

# 16) Flammable above Heat
RULES.append(Rule(
    rule_id="R16",
    description="Flammable item above heat source (drip/burn hazard).",
    match=lambda ctx: has(ctx['catA'], 'flammable') and has(ctx['catB'], 'heat') and ctx['rel'] == 'above',
    barrier=lambda ctx: above_barrier(ctx['posA'], ctx['posB'], z_min=0.2, r_align=0.3),
    risk_scale=0.15,
    weight=1.2,
))

# 17) Liquid above Human
RULES.append(Rule(
    rule_id="R17",
    description="Liquid above human (spill hazard).",
    match=lambda ctx: has(ctx['catA'], 'liquid') and has(ctx['catB'], 'human') and ctx['rel'] == 'above',
    barrier=lambda ctx: above_barrier(ctx['posA'], ctx['posB'], z_min=0.3, r_align=0.25),
    risk_scale=0.12,
    weight=1.0,
))

# 18) Robot near Fragile
RULES.append(Rule(
    rule_id="R18",
    description="Robot end-effector near fragile item (collision risk).",
    match=lambda ctx: has(ctx['catA'], 'robot') and has(ctx['catB'], 'fragile') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.2),
    risk_scale=0.1,
    weight=1.3,
))

# 19) Heavy near Human
RULES.append(Rule(
    rule_id="R19",
    description="Heavy object near human (impact hazard).",
    match=lambda ctx: ctx['objA'].mass and ctx['objA'].mass > 5.0 and has(ctx['catB'], 'human') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.3),
    risk_scale=0.15,
    weight=1.4,
))

# 20) Voltage near Human
RULES.append(Rule(
    rule_id="R20",
    description="Voltage source near human (shock hazard).",
    match=lambda ctx: has(ctx['catA'], 'voltage') and has(ctx['catB'], 'human') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.4),
    risk_scale=0.2,
    weight=1.6,
))

# 21) Flammable near Electronics
RULES.append(Rule(
    rule_id="R21",
    description="Flammable material near electronics (fire spread hazard).",
    match=lambda ctx: has(ctx['catA'], 'flammable') and has(ctx['catB'], 'electronics') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.3),
    risk_scale=0.15,
    weight=1.2,
))

# 22) Human near Heat
RULES.append(Rule(
    rule_id="R22",
    description="Human near heat source (burn hazard).",
    match=lambda ctx: has(ctx['catA'], 'human') and has(ctx['catB'], 'heat') and ctx['rel'] == 'near',
    barrier=lambda ctx: distance_barrier(ctx['posA'], ctx['posB'], d_min=0.4),
    risk_scale=0.12,
    weight=1.3,
))
