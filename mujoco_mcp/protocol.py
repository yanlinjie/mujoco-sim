# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validate the map and route evidence exchanged with MCP clients."""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlsplit


class ServiceError(RuntimeError):
    """A planning or simulation operation failed."""


def validate_endpoint(endpoint: str) -> None:
    """Require an HTTP service URL without credentials, query, or fragment."""
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("navigation service endpoints must be HTTP(S) base URLs")
    # Accessing port also validates malformed and out-of-range port numbers.
    _ = parsed.port


def finite_number(value: Any, field: str) -> float:
    """Read a finite JSON number, excluding booleans."""
    if type(value) not in (int, float):
        raise ServiceError(f"{field} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ServiceError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ServiceError(f"{field} must be a finite number")
    return number


def map_identity(response: dict[str, Any]) -> tuple[str, str]:
    """Require the map identity shared by planning and simulation responses."""
    for field in ("map_id", "map_version"):
        if not isinstance(response.get(field), str) or not response[field]:
            raise ServiceError(f"{field} must be a nonempty string")
    return response["map_id"], response["map_version"]


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
