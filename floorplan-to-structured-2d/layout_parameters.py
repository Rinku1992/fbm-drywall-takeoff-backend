CEILING_CHOICES = [
    "Flat",
    "Single-sloped",
    "Gable",
    "Tray",
    "Barrel vault",
    "Coffered",
    "Combination",
    "Soffit",
    "Cove",
    "Dome",
    "Cloister Vault",
    "Knee-Wall",
    "Cathedral with Flat Center",
    "Angled-Plane",
    "Boxed-Beam"
]

WALL_CHOICES = [
  {"wall_type": "OPEN_TO_BELOW", "is_height_static": False},
  {"wall_type": "FULL_WALL", "is_height_static": True},
  {"wall_type": "HALF_WALL", "is_height_static": False},
  {"wall_type": "STAIRCASE_WALL", "is_height_static": False},
  {"wall_type": "SOFFITS", "is_height_static": False},
  {"wall_type": "MULTI_FLOOR_ALIGNMENT", "is_height_static": False},
  {"wall_type": "DEMISING_WALL", "is_height_static": True},
  {"wall_type": "GARAGE_SEPARATION_WALL", "is_height_static": True},
  {"wall_type": "SHAFT_WALL", "is_height_static": False},
  {"wall_type": "WET_WALL", "is_height_static": False},
  {"wall_type": "HALLWAY_WALL", "is_height_static": True} 
]