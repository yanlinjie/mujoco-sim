# Copyright 2026 The RPent Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run Go2 simulation and its map viewer through one MCP service."""

from __future__ import annotations

import argparse
import math
import os
import re
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

from anyio import (
    CancelScope,
    Lock,
    fail_after,
    get_cancelled_exc_class,
    sleep,
    to_thread,
)
import uvicorn
from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import FileResponse, Response

from .planning import PlanningClient
from .protocol import (
    ServiceError,
    finite_number,
    map_identity,
    simulator_can_move,
    validate_walk_arrival,
)
from .simulation import Simulation


def _run_id(ctx: Context) -> str | None:
    meta = ctx.request_context.meta
    tag = meta.model_extra.get("rpent.dev/run-id") if meta is not None else None
    if tag is not None and (
        not isinstance(tag, str) or not re.fullmatch(r"[a-f0-9]{32}", tag)
    ):
        raise ValueError("invalid run identifier in MCP metadata")
    return tag


async def _owned_status(simulation: Simulation, walk_id: str) -> dict[str, Any]:
    status = await to_thread.run_sync(simulation.get_walk_status)
    if status.get("walk_id") != walk_id:
        raise ServiceError(f"simulation replaced walk {walk_id}")
    return status


async def _wait_for_walk(
    simulation: Simulation, walk_id: str, request: Request | None = None
) -> dict[str, Any]:
    while True:
        status = await _owned_status(simulation, walk_id)
        if status["state"] == "finished":
            return status
        if request is not None and await request.is_disconnected():
            raise ServiceError(f"caller disconnected during walk {walk_id}")
        await sleep(0.2)


async def _stop_walk(simulation: Simulation, walk_id: str) -> dict[str, Any]:
    with fail_after(15, shield=True):
        status = await _owned_status(simulation, walk_id)
        if status["state"] == "running":
            await to_thread.run_sync(simulation.stop_walk, walk_id)
            return await _wait_for_walk(simulation, walk_id)
        return status


def create_mujoco_mcp(
    simulation: Simulation,
    planning: PlanningClient,
    *,
    host: str,
    port: int,
    walk_timeout_s: float = 120.0,
) -> FastMCP:
    """Build a service that executes a saved A* MCP route by plan ID."""
    if not math.isfinite(walk_timeout_s) or walk_timeout_s <= 0:
        raise ValueError("walk timeout must be finite and positive")
    server = FastMCP(
        "unitree-mujoco",
        host=host,
        port=port,
        json_response=True,
    )
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
    walk_lock = Lock()
    submitted: set[str] = set()
    failed_previews: set[str] = set()
    reports: dict[str, dict[str, Any]] = {}

    async def load_plan(plan_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        route = await planning.read("get_plan", {"plan_id": plan_id})
        health = await to_thread.run_sync(simulation.health)
        if map_identity(route) != map_identity(health):
            raise ServiceError("saved plan uses a different map or map version")
        return route, health

    @server.tool(annotations=readonly)
    async def get_pose() -> dict[str, Any]:
        """Read the simulator's current x, y and yaw for A* planning."""
        return await to_thread.run_sync(simulation.get_pose)

    @server.tool(annotations=readonly)
    async def get_navigation_status() -> dict[str, Any]:
        """Check the live motion controller and whether its map matches A*."""
        health = await to_thread.run_sync(simulation.health)
        identity = map_identity(health)
        data = await planning.read("list_waypoints", {})
        return {
            "ready": simulator_can_move(health)
            and not walk_lock.locked()
            and not (health.get("control") or {}).get("requires_recovery")
            and identity == map_identity(data),
            "map_id": identity[0],
            "map_version": identity[1],
        }

    @server.tool(annotations=readonly)
    async def get_walk_status() -> dict[str, Any]:
        """Read the latest simulator walk and its execution evidence."""
        return await to_thread.run_sync(simulation.get_walk_status)

    @server.tool(annotations=readonly)
    async def preview_plan(plan_id: str) -> dict[str, Any]:
        """Preview a saved A* plan in an independent simulation."""
        async with walk_lock:
            route, _ = await load_plan(plan_id)
            failed_previews.add(plan_id)
            result = await to_thread.run_sync(
                partial(simulation.preview, route, timeout_s=walk_timeout_s)
            )
            preview = result.get("result")
            if result.get("ok") is not True or (
                isinstance(preview, dict)
                and (preview.get("ok") is False or any(
                    leg.get("reached") is False
                    for leg in preview.get("legs", [])
                    if isinstance(leg, dict)
                ))
            ):
                raise ServiceError("preview did not confirm arrival")
            failed_previews.discard(plan_id)
            return result

    async def execute_walk(
        plan_id: str, ctx: Context, evidence: dict[str, Any]
    ) -> dict[str, Any]:
        if plan_id in submitted:
            raise ServiceError("plan was already submitted; query the walk status")
        if plan_id in failed_previews:
            raise ServiceError("plan preview failed; replan before walking")
        route, health = await load_plan(plan_id)
        if not simulator_can_move(health):
            raise ServiceError("simulator is unavailable or busy")
        legs = route.get("legs")
        if not isinstance(legs, list) or len(legs) != 1:
            raise ServiceError("follow_plan requires a single-goal plan")
        pose = await to_thread.run_sync(simulation.get_pose)
        start = route["start"]
        x = finite_number(start.get("x"), "start.x")
        y = finite_number(start.get("y"), "start.y")
        if pose["fallen"] or math.hypot(pose["x"] - x, pose["y"] - y) > 0.3:
            raise ServiceError("the simulator moved away from the plan start or fell")
        walk_id = None
        try:
            with CancelScope(shield=True):
                submitted.add(plan_id)
                evidence["submission_uncertain"] = True
                accepted = await to_thread.run_sync(simulation.start_walk, route)
                walk_id = accepted["walk_id"]
                evidence.update(
                    walk_id=walk_id,
                    active_walk_id=walk_id,
                    submission_uncertain=False,
                )
            request = ctx.request_context.request
            try:
                with fail_after(walk_timeout_s):
                    status = await _wait_for_walk(
                        simulation,
                        walk_id,
                        request if isinstance(request, Request) else None,
                    )
            except TimeoutError as exc:
                raise ServiceError(
                    f"walk {walk_id} did not finish within {walk_timeout_s:g} seconds"
                ) from exc
            evidence.update(
                active_walk_id=None, stop_confirmed=True, pose=status.get("pose")
            )
            validate_walk_arrival(status, route)
            return {
                "ok": True,
                "navigation_completed": True,
                "plan_id": plan_id,
                "walk_id": walk_id,
                "state": "finished",
                "pose": status["pose"],
                "stop_confirmed": True,
                "result": status["result"],
            }
        finally:
            if evidence.get("active_walk_id"):
                assert walk_id is not None
                try:
                    stopped = await _stop_walk(simulation, walk_id)
                except Exception as exc:
                    evidence["stop_error"] = str(exc)
                    raise
                evidence.update(
                    active_walk_id=None, stop_confirmed=True, pose=stopped.get("pose")
                )

    @server.tool()
    async def follow_plan(plan_id: str, ctx: Context) -> dict[str, Any]:
        """Follow a plan_id returned by A* plan_route and wait for verified arrival."""
        run_id = _run_id(ctx)
        if walk_lock.locked():
            raise ServiceError("the simulation MCP service is already handling a plan")
        async with walk_lock:
            if reports.get(run_id, {}).get("requires_attention"):
                raise ServiceError("the previous walk requires reconciliation")
            evidence = {"plan_id": plan_id, "navigation_completed": False}
            report = {"state": "running", "evidence": {"navigation": evidence}}
            if run_id:
                reports.pop(run_id, None)
                reports[run_id] = report
                if len(reports) > 256:
                    del reports[next(iter(reports))]
            try:
                result = await execute_walk(plan_id, ctx, evidence)
                evidence["navigation_completed"] = True
                return result
            except get_cancelled_exc_class():
                raise
            except Exception as exc:
                report["error"] = str(exc)
                raise
            finally:
                report.update(
                    state="finished",
                    requires_attention=bool(
                        evidence.get("active_walk_id")
                        or evidence.get("submission_uncertain")
                        or evidence.get("stop_error")
                    ),
                    stop_error=evidence.get("stop_error"),
                )

    @server.tool(annotations=readonly)
    def get_run_report(ctx: Context) -> dict[str, Any]:
        """Read the execution and stop evidence associated with the caller's run."""
        return reports.get(_run_id(ctx), {"state": "missing"})

    return server


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default=os.environ.get("ROBOT_MAP", "room1110/2026-11-11.2"))
    parser.add_argument("--robot-root", type=Path, default=Path(os.environ.get("ROBOT_ROOT", root)))
    parser.add_argument("--output", type=Path, default=root / "output/robot-mujoco-local")
    parser.add_argument("--simulator-dir", type=Path, help="directory containing the built unitree_mujoco binary")
    parser.add_argument("--astar-mcp-url", default="http://127.0.0.1:8790/mcp")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8791, type=int)
    parser.add_argument("--dds-domain", default=1, type=int)
    parser.add_argument("--walk-timeout-s", default=120.0, type=float)
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--no-first-person", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.dds_domain <= 232:
        parser.error("--dds-domain must be between 0 and 232")
    output = args.output.resolve()
    robot_root = args.robot_root.resolve()
    map_dir = (robot_root / "slam_map_demo/maps" / args.map).resolve()
    simulator_dir = (args.simulator_dir or output / "native/build").resolve()
    simulation = Simulation(robot_root, map_dir, output, simulator_dir, args.dds_domain)
    server = create_mujoco_mcp(
        simulation, PlanningClient(args.astar_mcp_url), host=args.host,
        port=args.port, walk_timeout_s=args.walk_timeout_s,
    )

    @server.custom_route("/files/{name}", methods=["GET"])
    async def preview_image(request: Request) -> Response:
        name = request.path_params["name"]
        path = simulation.work_dir / name
        if not re.fullmatch(r"preview_[0-9]{6}_[0-9]+_[0-9]+\.png", name) or not path.is_file():
            return Response(status_code=404)
        return FileResponse(path, media_type="image/png")

    viewer = None
    if not args.no_viewer:
        from slam_map_demo.scripts.sim_stream import Viewer

        viewer = Viewer(simulation, first_person=not args.no_first_person)
        viewer.add_routes(server)

    app = server.streamable_http_app()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            await to_thread.run_sync(simulation.start)
            if viewer is not None:
                viewer.start()
            async with server.session_manager.run():
                yield
        finally:
            with CancelScope(shield=True):
                try:
                    if viewer is not None:
                        await to_thread.run_sync(viewer.close)
                finally:
                    await to_thread.run_sync(simulation.close)

    app.router.lifespan_context = lifespan
    uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
