# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loopback MCP servers backed by an in-memory simulation."""

from __future__ import annotations

import asyncio
import math
import socket
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP

from mujoco_mcp.planning import PlanningClient
from mujoco_mcp.server import create_mujoco_mcp
from mujoco_mcp.protocol import ServiceError


@pytest.fixture
def navigation_services():
    identity = {"map_id": "test_room", "map_version": "v1"}
    state = SimpleNamespace(
        submissions=[],
        pose={"x": 0.0, "y": 0.0, "yaw": 0.0, "fallen": False},
        sim_version="v1",
        running=False,
        hold=False,
        stopped=False,
        failed=False,
        simulator_running=True,
        other_controllers=[],
        remaining=0,
        polls_before_finish=1,
        walk_id=None,
        status_walk_id=None,
        submit_timeout=False,
        preview_failed=False,
        final_mutator=None,
        status_unavailable=False,
        started=threading.Event(),
    )
    waypoints = [
        {
            "id": "wp_01",
            "name": "广告牌",
            "stand_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0},
        },
        {
            "id": "wp_02",
            "name": "厕所",
            "stand_pose": {"x": 1.0, "y": 1.0, "yaw": 1.57},
        },
    ]

    class FakeSimulation:
        def health(self):
            return {
                "ok": True, **identity, "map_version": state.sim_version,
                "simulator_running": state.simulator_running,
                "other_controllers": state.other_controllers,
                "walk": {"state": "running" if state.running else "finished"},
            }

        def get_pose(self):
            return {"ok": True, **state.pose}

        def preview(self, route, *, timeout_s):
            return {"ok": True, "result": {"ok": not state.preview_failed}, "image": "/files/preview.png"}

        def start_walk(self, route):
            state.submissions.append(route)
            state.running, state.stopped, state.remaining = True, False, state.polls_before_finish
            state.walk_id = f"walk_{len(state.submissions)}"
            state.started.set()
            if state.submit_timeout:
                raise ServiceError("acceptance timed out")
            return {"ok": True, "walk_id": state.walk_id}

        def stop_walk(self, walk_id):
            assert walk_id == state.walk_id
            state.stopped = True

        def get_walk_status(self):
            if state.status_unavailable:
                raise ServiceError("status temporarily unavailable")
            walk_id = state.status_walk_id or state.walk_id
            if not state.stopped and (state.hold or state.remaining > 0):
                state.remaining -= 1
                return {"ok": True, "state": "running", "walk_id": walk_id}
            state.running = False
            reached = not state.failed and not state.stopped
            goal = state.submissions[-1]["legs"][0]["goal"]["stand_pose"]
            if reached:
                state.pose.update(x=goal["x"], y=goal["y"], yaw=goal.get("yaw") or 0.0)
            result = {
                "ok": True, "state": "finished", "walk_id": walk_id, "fell": False,
                "pose": state.pose.copy(),
                "result": {"ok": reached, "legs": [{
                    "reached": reached, "position_error": 0.0 if reached else 1.0,
                    "yaw_error": 0.0, "time": 1.0,
                }]},
            }
            if state.final_mutator:
                state.final_mutator(result)
            return result

    return SimpleNamespace(state=state, identity=identity, waypoints=waypoints, sim=FakeSimulation())


@contextmanager
def run_mcp(mcp):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(mcp.streamable_http_app(), log_level="warning")
        )
        worker = threading.Thread(
            target=server.run, kwargs={"sockets": [listener]}, daemon=True
        )
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started:
                assert time.monotonic() < deadline and worker.is_alive()
                time.sleep(0.01)
            yield f"http://127.0.0.1:{port}/mcp"
        finally:
            server.should_exit = True
            worker.join(5)
            assert not worker.is_alive()


class LoopbackClient:
    def __init__(self, url):
        self.url = url

    async def invoke(self, name, arguments, *, meta=None, terminate_on_close=True):
        async with httpx.AsyncClient(trust_env=False, timeout=15) as http:
            async with streamable_http_client(
                self.url, http_client=http, terminate_on_close=terminate_on_close,
            ) as (
                read,
                write,
                _,
            ):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(name, arguments, meta=meta)

    def call_tool(self, name, arguments, *, meta=None):
        return asyncio.run(self.invoke(name, arguments, meta=meta))


@pytest.fixture
def clients(navigation_services):
    plans = {}
    astar = FastMCP("astar-test", stateless_http=True, json_response=True)
    identity = navigation_services.identity
    waypoints = navigation_services.waypoints

    @astar.tool()
    def list_waypoints() -> dict[str, Any]:
        return {"ok": True, **identity, "waypoints": waypoints}

    @astar.tool()
    def plan_route(start: dict, goal: str | dict) -> dict[str, Any]:
        target = (
            next(p for p in waypoints if goal in (p["name"], p["id"]))
            if isinstance(goal, str) else
            {"id": "xy", "name": "coordinate", "stand_pose": goal}
        )
        pose = target["stand_pose"]
        dx, dy = pose["x"] - start["x"], pose["y"] - start["y"]
        length = math.hypot(dx, dy)
        plan_id = uuid4().hex
        plans[plan_id] = {
            "ok": True, **identity, "frame": "map",
            "units": {"length": "meter", "angle": "radian"},
            "start": start, "goals": [target["id"]], "total_length": length,
            "legs": [{
                "query": goal, "goal": target, "length": length,
                "path": [[start["x"], start["y"]], [pose["x"], pose["y"]]],
                "headings": [math.atan2(dy, dx)],
            }],
        }
        return {"ok": True, "plan_id": plan_id}

    @astar.tool()
    def get_plan(plan_id: str) -> dict[str, Any]:
        return plans[plan_id]

    with run_mcp(astar) as astar_url:
        simulator = create_mujoco_mcp(
            navigation_services.sim, PlanningClient(astar_url),
            host="127.0.0.1", port=0, walk_timeout_s=1.5,
        )
        with run_mcp(simulator) as sim_url:
            yield LoopbackClient(astar_url), LoopbackClient(sim_url)
