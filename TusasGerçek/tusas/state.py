from typing import Dict

from .zones.manager import ZoneManager


# In-memory session store: session_id -> ZoneManager
zone_managers = {}  # type: Dict[str, ZoneManager]


def get_zone_manager(session_id: str) -> ZoneManager:
    return zone_managers.get(session_id)


def set_zone_manager(session_id: str, manager: ZoneManager) -> None:
    zone_managers[session_id] = manager

