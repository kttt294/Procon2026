"""
HTTP client for communicating with the contest server.

Endpoints (assumed, update when BTC publishes protocol):
  GET  /match-config          → MatchConfig + MapData
  GET  /state/day/{d}         → DayState
  POST /action/day/{d}        → submit DayOrders
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import time
from typing import List, Optional

import requests

from env.models import (
    AgentAction, AgentState, Cell, CMD_MOVE, CMD_STAY,
    DayOrder, DayState, MapData, MatchConfig, Spot,
)


class ContestClient:
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base    = base_url.rstrip("/")
        self.timeout = timeout_s
        self.session = requests.Session()

    # ------------------------------------------------------------------ #
    # Phase 0: pre-match                                                   #
    # ------------------------------------------------------------------ #

    def get_match_config(self) -> tuple[MatchConfig, MapData, List[AgentState]]:
        resp = self._get("/match-config")
        cfg      = MatchConfig.from_dict(resp)
        map_data = MapData.from_dict(resp)
        agents   = [
            AgentState(
                id=a["id"],
                type=a["type"],
                cell=a["start_cell"],
                fuel=a.get("fuel_capacity", 0),
            )
            for a in resp["my_agents"]
        ]
        return cfg, map_data, agents

    # ------------------------------------------------------------------ #
    # Phase 1: per-day                                                     #
    # ------------------------------------------------------------------ #

    def get_day_state(
        self,
        day: int,
        cfg: MatchConfig,
        map_data: MapData,
        prev_state: Optional[DayState] = None,
    ) -> DayState:
        resp = self._get(f"/state/day/{day}")
        return self._parse_state(resp, cfg, map_data, prev_state)

    def submit_orders(self, day: int, orders: List[DayOrder], timeout_s: Optional[float] = None) -> dict:
        payload = {"orders": [o.to_dict() for o in orders]}
        resp = self._post(f"/action/day/{day}", payload, timeout_s)
        return resp

    def submit_with_retry(
        self,
        day: int,
        orders: List[DayOrder],
        fallback_orders: List[DayOrder],
        deadline_ms: int,
        start_ms: float,
    ) -> dict:
        """
        Try to submit orders. If invalid, retry with fallback.
        start_ms is a time.monotonic() timestamp in seconds (legacy name).
        Keep 50 ms for response handling; at most two attempts.
        """
        for attempt, o in enumerate([orders, fallback_orders]):
            remaining = start_ms + deadline_ms / 1000 - time.monotonic() - .05
            if remaining <= 0:
                break
            try:
                resp = self.submit_orders(day, o, timeout_s=remaining)
                if resp.get("status") == "valid":
                    return {**resp, "_accepted_orders": o}
            except Exception as e:
                print(f"[client] submit attempt {attempt+1} failed: {e}")
        return {"status": "failed"}

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def _parse_state(
        self,
        resp: dict,
        cfg: MatchConfig,
        map_data: MapData,
        prev: Optional[DayState],
    ) -> DayState:
        traffic = {r["cell_id"]: r["status"] for r in resp.get("traffic", [])}
        inventory = {s["cell_id"]: s["inventory"] for s in resp.get("spots", [])}

        agents = []
        for a in resp.get("my_agents", []):
            agents.append(AgentState(
                id=a["id"],
                type=a["type"],
                cell=a["cell"],
                fuel=a.get("fuel", 0),
            ))

        opponents = [o["cell"] for o in resp.get("opponents", [])]

        # Score bookkeeping carried from prev state (server may also send it)
        score = resp.get("my_score", {})
        collected = set(score.get("unique_series", []))
        total_udon = score.get("total_udon", 0)
        daily_hist = [set(d) for d in score.get("daily_series_history", [])]

        if prev is not None and not collected:
            collected  = set(prev.collected_series)
            total_udon = prev.total_udon
            daily_hist = list(prev.daily_series)

        return DayState(
            day=resp["day"],
            steps_left=resp["steps_left"],
            time_limit_ms=resp.get("time_limit_ms", 5000),
            traffic=traffic,
            spot_inventory=inventory,
            my_agents=agents,
            opponent_cells=opponents,
            collected_series=collected,
            daily_series=daily_hist,
            total_udon=total_udon,
        )

    # ------------------------------------------------------------------ #
    # HTTP helpers                                                         #
    # ------------------------------------------------------------------ #

    def _get(self, path: str) -> dict:
        url = self.base + path
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict, timeout_s: Optional[float] = None) -> dict:
        url = self.base + path
        timeout = self.timeout if timeout_s is None else min(self.timeout, timeout_s)
        if timeout <= 0:
            raise TimeoutError("Submission deadline expired")
        resp = self.session.post(url, json=body, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
