"""基于碰撞高度场的二维栅格工具：障碍判定、离障碍距离、A* 路径规划。

点位标注和导航脚本共用，保证“标注时认为能走”和“导航时认为能走”使用同一套标准。
"""
import heapq

import numpy as np
from scipy import ndimage

if __package__:
    from .map_to_mujoco import FLOOR_MAX_HEIGHT
else:
    from map_to_mujoco import FLOOR_MAX_HEIGHT

OBSTACLE_HEIGHT = FLOOR_MAX_HEIGHT  # 高于此值视为障碍；地毯（6–9 cm）可以走上去
ROBOT_RADIUS = 0.30      # 规划时的硬性避障半径（Go2 半宽约 0.2 m，加余量）
PREFERRED_CLEARANCE = 0.6  # 离障碍小于此距离时增加代价，让路线尽量走在通道中间


class NavGrid:
    def __init__(self, heights, lo, resolution, observed):
        self.heights = heights
        self.lo = np.asarray(lo, dtype=float)
        self.res = float(resolution)
        # 没有扫描数据的区域和地图边界一律视为障碍：那里可能是没扫到的墙，也可能是地图外
        passable = (heights <= OBSTACLE_HEIGHT) & observed
        passable[[0, -1], :] = passable[:, [0, -1]] = False
        self.clearance = ndimage.distance_transform_edt(passable) * self.res

    @classmethod
    def load(cls, npz_path):
        data = np.load(npz_path)
        return cls(data["heights"], data["lo"], data["resolution"], data["observed"])

    def to_cell(self, xy):
        c, r = np.round((np.asarray(xy) - self.lo) / self.res).astype(int)
        return int(r), int(c)

    def to_xy(self, cell):
        r, c = cell
        return self.lo + np.array([c, r]) * self.res

    def inside(self, cell):
        r, c = cell
        return 0 <= r < self.heights.shape[0] and 0 <= c < self.heights.shape[1]

    def clearance_at(self, xy):
        cell = self.to_cell(xy)
        return self.clearance[cell] if self.inside(cell) else 0.0

    def segment_free(self, a, b, radius=ROBOT_RADIUS):
        """线段上每隔半个栅格采样，检查离障碍距离都大于 radius。"""
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(int(np.linalg.norm(b - a) / (self.res / 2)), 1)
        return all(self.clearance_at(a + (b - a) * t) > radius for t in np.linspace(0, 1, n + 1))

    def plan(self, start, goal):
        """A* 搜索，返回经过拉直的折线（地图坐标列表）；找不到返回 None。"""
        start_cell, goal_cell = self.to_cell(start), self.to_cell(goal)
        free = self.clearance > ROBOT_RADIUS
        prefix = []
        if not self.inside(start_cell) or self.clearance[start_cell] <= 0:
            # 起点在地图外、未重建区域或障碍里（例如机器狗被推出了地图）：先直线走回最近的可通行栅格
            esc = self._escape_cell(start, free, goal_cell)
            if esc is None:
                return None
            prefix, start_cell = [np.asarray(start, float)], esc
            start = self.to_xy(esc)
        # 起点/终点自身允许略贴近障碍（例如站位紧挨家具），只要求不在障碍里
        free[start_cell] = free[goal_cell] = True
        penalty = np.clip(PREFERRED_CLEARANCE - self.clearance, 0, None) * 8.0

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        g = {start_cell: 0.0}
        parent = {}
        heap = [(0.0, start_cell)]
        goal_xy = self.to_xy(goal_cell)
        while heap:
            _, cell = heapq.heappop(heap)
            if cell == goal_cell:
                break
            for dr, dc in moves:
                nxt = (cell[0] + dr, cell[1] + dc)
                if not self.inside(nxt) or not free[nxt]:
                    continue
                step = np.hypot(dr, dc) * self.res
                cost = g[cell] + step * (1 + penalty[nxt])
                if cost < g.get(nxt, np.inf):
                    g[nxt] = cost
                    parent[nxt] = cell
                    h = np.linalg.norm(self.to_xy(nxt) - goal_xy)
                    heapq.heappush(heap, (cost + h, nxt))
        if goal_cell not in g:
            return None

        cells = [goal_cell]
        while cells[-1] != start_cell:
            cells.append(parent[cells[-1]])
        points = [self.to_xy(c) for c in reversed(cells)]
        points[0], points[-1] = np.asarray(start, float), np.asarray(goal, float)
        return prefix + self._shortcut(points)

    def _escape_cell(self, start, free, goal_cell):
        """离 start 最近、与终点连通、且直线过去不穿过实际障碍（高于 OBSTACLE_HEIGHT 的栅格）的可通行栅格。
        未重建区域在仿真中是平地，可以穿过。"""
        labels, _ = ndimage.label(free)
        target = labels[goal_cell] if self.inside(goal_cell) else 0
        rows, cols = np.nonzero(labels == target) if target else np.nonzero(free)
        xy = self.lo + np.stack([cols, rows], axis=1) * self.res
        start = np.asarray(start, float)
        for k in np.argsort(np.linalg.norm(xy - start, axis=1))[:400]:
            n = max(int(np.linalg.norm(xy[k] - start) / (self.res / 2)), 1)
            cells = [self.to_cell(start + (xy[k] - start) * t) for t in np.linspace(0, 1, n + 1)]
            if all(not self.inside(c) or self.heights[c] <= OBSTACLE_HEIGHT for c in cells):
                return int(rows[k]), int(cols[k])
        return None

    def _shortcut(self, points):
        """贪心拉直：从当前点出发，跳到最远的、直线可达且离障碍足够远的路径点。"""
        result = [points[0]]
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.segment_free(points[i], points[j], PREFERRED_CLEARANCE * 0.75):
                j -= 1
            result.append(points[j])
            i = j
        return result
