# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Client for the external MuJoCo route-following service."""

from __future__ import annotations

import math
from typing import Any

from .http_client import (
    JsonServiceClient,
    ServiceError,
    finite_number,
    map_identity,
)


def validate_pose(pose: Any) -> None:
    """Require finite map-frame position and heading observations."""
    if not isinstance(pose, dict):
        raise ServiceError("simulation pose must be an object")
    for key in ("x", "y", "yaw"):
        finite_number(pose.get(key), f"pose.{key}")
    if type(pose.get("fallen")) is not bool:
        raise ServiceError("pose.fallen must be boolean")


def simulator_can_move(health: dict[str, Any]) -> bool:
    """Check whether the simulator has an idle motion controller."""
    return (
        health["simulator_running"]
        and health.get("control_mode", "idle") == "idle"
        and not health.get("other_controllers")
        and (health.get("walk") or {}).get("state") != "running"
    )


def validate_walk_arrival(status: dict[str, Any], plan: dict[str, Any]) -> None:
    """Verify the service's single-goal result against the observed final pose."""
    result = status.get("result")
    if (
        status.get("fell") is not False
        or not isinstance(result, dict)
        or result.get("ok") is not True
        or not isinstance(result.get("legs"), list)
        or len(result["legs"]) != 1
        or not isinstance(result["legs"][0], dict)
        or result["legs"][0].get("reached") is not True
    ):
        raise ServiceError("walk finished without confirmed arrival at the goal")
    leg = result["legs"][0]
    position_error = finite_number(leg.get("position_error"), "position_error")
    if not 0 <= position_error <= 0.25 + 1e-9:
        raise ServiceError("navigation position error exceeds 0.25 m")
    pose = status.get("pose")
    validate_pose(pose)
    if pose["fallen"]:
        raise ServiceError("robot has fallen")
    goal = plan["legs"][0]["goal"]["stand_pose"]
    if math.hypot(pose["x"] - goal["x"], pose["y"] - goal["y"]) > 0.25 + 1e-9:
        raise ServiceError("final pose is outside the navigation goal tolerance")
    if goal.get("yaw") is not None:
        yaw_error = finite_number(leg.get("yaw_error"), "yaw_error")
        actual_error = math.atan2(
            math.sin(pose["yaw"] - goal["yaw"]),
            math.cos(pose["yaw"] - goal["yaw"]),
        )
        if max(abs(yaw_error), abs(actual_error)) > 0.2 + 1e-9:
            raise ServiceError("navigation yaw error exceeds 0.2 radians")


class SimulationClient(JsonServiceClient):
    """Submit routes and query asynchronous walks, without managing simulators."""

    def health(self) -> dict[str, Any]:
        """Read map identity, simulator availability, and active walk state."""
        result = self.request("GET", "/health")
        map_identity(result)
        if type(result.get("simulator_running")) is not bool:
            raise ServiceError("simulator_running must be boolean")
        return result

    def get_pose(self) -> dict[str, Any]:
        """Read a fresh pose; callers decide whether a fallen robot may proceed."""
        result = self.request("GET", "/pose")
        validate_pose(result)
        return result

    def preview(self, plan: dict[str, Any], *, timeout_s: float) -> dict[str, Any]:
        """Run the service's independent preview without moving the live simulator."""
        return self.request("POST", "/preview", plan, timeout_s=timeout_s)

    def start_walk(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Submit once; an accepted response is not a completed walk."""
        result = self.request("POST", "/walk", plan)
        if not isinstance(result.get("walk_id"), str) or not result["walk_id"]:
            raise ServiceError("accepted walk must contain a nonempty walk_id")
        return result

    def get_walk_status(self) -> dict[str, Any]:
        """Read a running or finished walk without inferring arrival success."""
        result = self.request("GET", "/walk")
        if result.get("state") not in ("running", "finished"):
            raise ServiceError("walk state must be running or finished")
        return result

    def stop_walk(self, walk_id: str) -> dict[str, Any]:
        """Request a stop; the caller must subsequently confirm finished state."""
        return self.request("POST", "/walk/stop", {"walk_id": walk_id})
