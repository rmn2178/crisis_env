"""Scenario generator for the Crisis Response Environment.

Mirrors issue_generator.py exactly in structure: templates, generation
functions, and ground-truth stripping.
"""

from __future__ import annotations

import copy
import hashlib
from random import Random
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Threat templates — at least 21 covering all required categories
# ---------------------------------------------------------------------------

THREAT_TEMPLATES: List[Dict[str, Any]] = [
    # ── AIRSTRIKE × CRITICAL (3) ──────────────────────────────────────────
    {
        "threat_type": "AIRSTRIKE",
        "location": "Military Base Alpha",
        "description": "Multiple inbound cruise missiles detected on radar. Smoke visible on perimeter. Personnel sheltering in hardened bunkers.",
        "time_to_impact": 30,
        "population_at_risk": 50,
        "has_visible_signal": True,
        "affected_systems": ["radar_array", "comm_tower", "runway"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-fighter_jet-01",
    },
    {
        "threat_type": "AIRSTRIKE",
        "location": "Forward Operating Base Delta",
        "description": "Explosion reported at ammunition depot. Secondary detonations imminent. Fire spreading toward barracks.",
        "time_to_impact": 25,
        "population_at_risk": 50,
        "has_visible_signal": True,
        "affected_systems": ["ammo_depot", "barracks", "fuel_storage"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-fighter_jet-01",
    },
    {
        "threat_type": "AIRSTRIKE",
        "location": "Air Defense Command Post",
        "description": "Hostile aircraft formation breached no-fly zone. SAM sites offline. Smoke visible from initial strafing run.",
        "time_to_impact": 20,
        "population_at_risk": 50,
        "has_visible_signal": True,
        "affected_systems": ["sam_battery", "command_center", "power_grid"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-fighter_jet-01",
    },
    # ── AIRSTRIKE × HIGH (3) ──────────────────────────────────────────────
    {
        "threat_type": "AIRSTRIKE",
        "location": "Logistics Depot Bravo",
        "description": "Unidentified aircraft observed circling supply route. No engagement yet. Heightened alert status.",
        "time_to_impact": 120,
        "population_at_risk": 30,
        "has_visible_signal": False,
        "affected_systems": ["supply_route", "vehicle_park"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-fighter_jet-01",
    },
    {
        "threat_type": "AIRSTRIKE",
        "location": "Training Camp Echo",
        "description": "Radar contact inbound from hostile territory. ETA uncertain. Personnel alert issued.",
        "time_to_impact": 90,
        "population_at_risk": 40,
        "has_visible_signal": False,
        "affected_systems": ["training_grounds", "control_tower"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-fighter_jet-01",
    },
    {
        "threat_type": "AIRSTRIKE",
        "location": "Rear Garrison Foxtrot",
        "description": "Intelligence intercept indicates potential air sortie against garrison. Defensive posture ordered.",
        "time_to_impact": 150,
        "population_at_risk": 35,
        "has_visible_signal": False,
        "affected_systems": ["garrison_perimeter", "motor_pool"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-fighter_jet-01",
    },
    # ── SHIP_ATTACK × CRITICAL (3) ────────────────────────────────────────
    {
        "threat_type": "SHIP_ATTACK",
        "location": "Port Vessel MV-7",
        "description": "Vessel taking on water after torpedo impact. SOS signal transmitted. Listing 15 degrees to starboard.",
        "time_to_impact": 45,
        "population_at_risk": 200,
        "has_visible_signal": True,
        "affected_systems": ["hull_integrity", "engine_room", "bridge"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-naval_vessel-01",
    },
    {
        "threat_type": "SHIP_ATTACK",
        "location": "Cargo Freighter SS-Oceanus",
        "description": "Fire in engine room after mine strike. Crew abandoning lower decks. SOS signal active.",
        "time_to_impact": 40,
        "population_at_risk": 200,
        "has_visible_signal": True,
        "affected_systems": ["engine_room", "cargo_hold", "life_rafts"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-naval_vessel-01",
    },
    {
        "threat_type": "SHIP_ATTACK",
        "location": "Ferry NV-Poseidon",
        "description": "Hostile fast boat rammed ferry hull. Breach below waterline. 200 passengers aboard. SOS signal broadcasting.",
        "time_to_impact": 35,
        "population_at_risk": 200,
        "has_visible_signal": True,
        "affected_systems": ["passenger_deck", "hull", "navigation"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-naval_vessel-01",
    },
    # ── SHIP_ATTACK × HIGH (3) ────────────────────────────────────────────
    {
        "threat_type": "SHIP_ATTACK",
        "location": "Patrol Boat PB-12",
        "description": "Hostile vessel shadowing patrol route. No engagement. Possible mine-laying activity detected.",
        "time_to_impact": 180,
        "population_at_risk": 80,
        "has_visible_signal": False,
        "affected_systems": ["patrol_route", "sonar_array"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-naval_vessel-01",
    },
    {
        "threat_type": "SHIP_ATTACK",
        "location": "Supply Ship SS-Hermes",
        "description": "Suspicious submarine contact near convoy lane. No torpedo launch detected yet.",
        "time_to_impact": 200,
        "population_at_risk": 100,
        "has_visible_signal": False,
        "affected_systems": ["convoy_lane", "supply_chain"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-naval_vessel-01",
    },
    {
        "threat_type": "SHIP_ATTACK",
        "location": "Research Vessel RV-Galileo",
        "description": "Unidentified craft approaching research vessel in contested waters. Crew on alert.",
        "time_to_impact": 160,
        "population_at_risk": 60,
        "has_visible_signal": False,
        "affected_systems": ["lab_deck", "communications"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-naval_vessel-01",
    },
    # ── DRONE_THREAT × CRITICAL (3) ───────────────────────────────────────
    {
        "threat_type": "DRONE_THREAT",
        "location": "Central Shopping Mall — Sector 9",
        "description": "Unauthorized UAV swarm detected over crowded mall. Drones carrying unidentified payload. Panic spreading among 1000 shoppers.",
        "time_to_impact": 50,
        "population_at_risk": 1000,
        "has_visible_signal": True,
        "affected_systems": ["air_space", "crowd_control", "cctv_network"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-ground_unit-01",
    },
    {
        "threat_type": "DRONE_THREAT",
        "location": "City Stadium — Main Arena",
        "description": "Drone spotted over packed stadium during event. Explosive signature detected by sensors. Evacuation not yet started.",
        "time_to_impact": 40,
        "population_at_risk": 1000,
        "has_visible_signal": True,
        "affected_systems": ["stadium_airspace", "spectator_zones", "exits"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-ground_unit-01",
    },
    {
        "threat_type": "DRONE_THREAT",
        "location": "Government District — Parliament Square",
        "description": "Multiple drones converging on parliament during session. Counter-drone systems jammed. Civilian crowd nearby.",
        "time_to_impact": 30,
        "population_at_risk": 1000,
        "has_visible_signal": True,
        "affected_systems": ["government_buildings", "public_square", "comms"],
        "_correct_priority": "CRITICAL",
        "_correct_resource": "RES-ground_unit-01",
    },
    # ── DRONE_THREAT × HIGH (3) ───────────────────────────────────────────
    {
        "threat_type": "DRONE_THREAT",
        "location": "Industrial Park — Zone C",
        "description": "Unauthorized UAV observed surveilling chemical plant. No payload detected yet.",
        "time_to_impact": 300,
        "population_at_risk": 150,
        "has_visible_signal": False,
        "affected_systems": ["chemical_storage", "perimeter_fence"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-ground_unit-01",
    },
    {
        "threat_type": "DRONE_THREAT",
        "location": "Residential Area — Block 14",
        "description": "Single drone flying erratic patterns over apartment complex. Residents alarmed.",
        "time_to_impact": 250,
        "population_at_risk": 200,
        "has_visible_signal": False,
        "affected_systems": ["residential_block", "local_police"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-ground_unit-01",
    },
    {
        "threat_type": "DRONE_THREAT",
        "location": "Highway Interchange — Junction 7",
        "description": "Drone spotted near major highway interchange. Traffic cameras show hovering above fuel tanker lane.",
        "time_to_impact": 200,
        "population_at_risk": 300,
        "has_visible_signal": False,
        "affected_systems": ["highway_traffic", "fuel_lane"],
        "_correct_priority": "HIGH",
        "_correct_resource": "RES-ground_unit-01",
    },
    # ── MEDIUM / LOW distractors (3) ──────────────────────────────────────
    {
        "threat_type": "DRONE_THREAT",
        "location": "Rural Farmland — Sector 22",
        "description": "Commercial agricultural drone off-course. Farmer reports seeing it near barn. No threat indicators.",
        "time_to_impact": 600,
        "population_at_risk": 5,
        "has_visible_signal": False,
        "affected_systems": ["farm_equipment"],
        "_correct_priority": "LOW",
        "_correct_resource": "RES-ground_unit-01",
    },
    {
        "threat_type": "AIRSTRIKE",
        "location": "Decommissioned Airstrip Zulu",
        "description": "Old radar ghost detected on obsolete equipment. No visual confirmation. Likely equipment malfunction.",
        "time_to_impact": 900,
        "population_at_risk": 2,
        "has_visible_signal": False,
        "affected_systems": ["legacy_radar"],
        "_correct_priority": "LOW",
        "_correct_resource": "RES-fighter_jet-01",
    },
    {
        "threat_type": "SHIP_ATTACK",
        "location": "Fishing Trawler FT-Sunrise",
        "description": "Fishing vessel reports minor hull scrape from floating debris. No immediate danger. Requesting inspection.",
        "time_to_impact": 1200,
        "population_at_risk": 10,
        "has_visible_signal": False,
        "affected_systems": ["hull_minor"],
        "_correct_priority": "MEDIUM",
        "_correct_resource": "RES-naval_vessel-01",
    },
]

# ---------------------------------------------------------------------------
# All available resource IDs in the environment
# ---------------------------------------------------------------------------

ALL_RESOURCES: List[str] = [
    "RES-fighter_jet-01",
    "RES-naval_vessel-01",
    "RES-ground_unit-01",
    "RES-medic_team-01",
    "RES-evacuation_unit-01",
]

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _stable_hash(s: str) -> int:
    """Deterministic hash for seeding."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def generate_threat(
    template: Dict[str, Any],
    threat_id: str,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """Create a concrete threat instance from a template.

    The returned dict contains both public fields (visible to the agent)
    and private ground-truth fields (prefixed with ``_correct_``).
    """
    rng = Random(_stable_hash(threat_id) + seed_offset)

    # Small deterministic jitter on time_to_impact
    jitter = rng.randint(-5, 5)
    time_to_impact = max(10, template["time_to_impact"] + jitter)

    # Small deterministic jitter on population
    pop_jitter = rng.randint(-2, 5)
    population_at_risk = max(1, template["population_at_risk"] + pop_jitter)

    # Severity string derived from correct priority
    severity = template["_correct_priority"]

    return {
        "threat_id": threat_id,
        "threat_type": template["threat_type"],
        "location": template["location"],
        "description": template["description"],
        "severity": severity,
        "time_to_impact": time_to_impact,
        "population_at_risk": population_at_risk,
        "has_visible_signal": template["has_visible_signal"],
        "affected_systems": list(template["affected_systems"]),
        "_correct_priority": template["_correct_priority"],
        "_correct_resource": template["_correct_resource"],
    }


def generate_scenario(task_difficulty: str) -> List[Dict[str, Any]]:
    """Generate a complete deterministic scenario for the given difficulty.

    Returns:
        A list of threat dicts (with ground-truth fields included).
    """
    if task_difficulty == "easy":
        indices = [0, 3, 6, 9, 12, 15][:5]  # 5 threats, clear spread
        threats: List[Dict[str, Any]] = []
        for i, idx in enumerate(indices):
            tid = f"THR-{1000 + i}"
            threats.append(generate_threat(THREAT_TEMPLATES[idx], tid, seed_offset=0))
        return threats

    elif task_difficulty == "medium":
        # 10 threats — pick a diverse spread plus a hidden cascade pair
        indices = [0, 3, 6, 9, 12, 15, 1, 4, 7, 10]
        threats = []
        for i, idx in enumerate(indices):
            tid = f"THR-{2000 + i}"
            threats.append(generate_threat(THREAT_TEMPLATES[idx], tid, seed_offset=1))

        # Hidden cascade: if THR-2003 (index 3) is resolved incorrectly,
        # THR-2007 (index 7) worsens.  We embed this by flagging both.
        threats[3]["_cascade_target"] = "THR-2007"
        threats[7]["_cascade_source"] = "THR-2003"
        threats[7]["_cascade_severity_bump"] = "CRITICAL"
        return threats

    else:  # hard
        # 13 base threats from across all templates
        base_indices = [0, 1, 2, 3, 6, 7, 8, 9, 12, 13, 14, 15, 16]
        threats = []
        for i, idx in enumerate(base_indices):
            tid = f"THR-{3000 + i}"
            threats.append(generate_threat(THREAT_TEMPLATES[idx], tid, seed_offset=2))

        # 2 subtle hidden cascades as extra threats
        extra_template_a = THREAT_TEMPLATES[17]  # HIGH drone
        extra_a = generate_threat(extra_template_a, "THR-3013", seed_offset=2)
        extra_a["_cascade_target"] = "THR-3002"
        extra_a["_cascade_severity_bump"] = "CRITICAL"

        extra_template_b = THREAT_TEMPLATES[18]  # HIGH drone
        extra_b = generate_threat(extra_template_b, "THR-3014", seed_offset=2)
        extra_b["_cascade_target"] = "THR-3008"
        extra_b["_cascade_severity_bump"] = "CRITICAL"

        threats.append(extra_a)
        threats.append(extra_b)

        # Mark cascade sources on targets
        for t in threats:
            if t["threat_id"] == "THR-3002":
                t["_cascade_source"] = "THR-3013"
            if t["threat_id"] == "THR-3008":
                t["_cascade_source"] = "THR-3014"

        # Shuffle deterministically
        Random(99).shuffle(threats)
        return threats


def strip_ground_truth(threat: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *threat* with all ``_correct_*`` and internal keys removed."""
    return {
        k: copy.deepcopy(v)
        for k, v in threat.items()
        if not k.startswith("_")
    }
