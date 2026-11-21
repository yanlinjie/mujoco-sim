# 三维场景图（Scene Graph）

把地图包里的几何（碰撞高度场、点云）、建图照片和点位库组织成一张分层的图，供导航算法查询：
- 给 **A\* 导航**：文字或编号 → 目标站位 (x, y, 朝向) → 在碰撞高度场上规划路线；
- 给 **视觉语言导航（VLN）/ 大语言模型规划**：整个场景的文字描述、按语义检索物体、
  沿通行网络的分段子目标（先到哪个地点、经过哪些区域）。

点位库 `waypoints.yaml` 是场景图的一层，不需要另外维护。

## 结构

| 层 | 节点 | 来源 | 本地图数量 |
|---|---|---|---|
| 区域 region | 主通道、丁字路口、北侧通道、东侧通道 | 配置文件给名称和种子点；地点按沿通行网络的距离归到最近的种子 | 4 |
| 地点 place | 机器狗能站立的位置（沿通道中线约每 0.5 m 一个） | 碰撞高度场中离障碍 > 0.3 m、与原点连通的区域，取离障碍距离的“山脊”（通道中线） | 15 |
| 物体 object | 货架、纸箱、木箱、手推车、机器人、柱子、标定板…… | 开放词汇检测器 Grounding DINO 在 99 张建图照片里检测，借地图点云定出三维位置，多张照片合并；中文 CLIP 复核类别并存图像向量 | 26 |
| 点位 waypoint | 点位库中的 7 个点位 | `waypoints.yaml` | 7 |

| 边 | 含义 |
|---|---|
| `traversable` 地点–地点 | 两地点间直线可通行（离障碍 > 0.3 m），带长度；地点和这些边构成**通行网络** |
| `in_region` 地点/物体/点位 → 区域 | 属于哪个区域 |
| `near` 物体 → 地点 | 到达该物体时的站位所在地点，带站位离物体表面的距离 |
| `at` 点位 → 地点 | 点位所在地点 |
| `next_to` 物体–物体 | 水平距离 < 0.3 m |
| `on` 物体 → 物体 | 放在另一个物体上面 |

每个物体还记录：中心、包围盒、尺寸、检测次数和视角数、类别得分前三名、裁剪图、是否可能移动（机器人），
以及**到达站位** `approach`：可达区域中离物体表面最近、且离障碍不少于 0.45 m（能原地转身）的位置，朝向物体。

## 文件

```
scripts/scene_graph/
├── build_scene_graph.py        # 构建（离线，需 torch + transformers）
├── scene_graph.py              # 运行时查询库（只依赖 numpy，可直接在导航进程里 import）
├── query_scene_graph.py        # 命令行查询
├── navigate_to.py              # 文字 → 场景图 → A* → 机器狗走过去（DDS 或无窗口）
└── configs/r2b_storage_ns.yaml # 本地图的区域名称、物体词表、检测参数

<地图包>/scene_graph/
├── scene_graph.json            # 节点和边（地图坐标系，米/弧度）
├── object_features.npy         # 物体的中文 CLIP 图像向量 (物体数, 768)，行号即节点的 feature_row
├── crops/obj_XX.jpg            # 每个物体的代表性裁剪图
└── overview.png                # 俯视示意图：区域着色、地点与通行边、物体框、点位
```

## 构建

```bash
R=/home/oc/projects/robot
D=$R/slam_map_demo
MAP=$D/maps/r2b_storage_ns/2026-11-01.3
cd /tmp
# 基础 conda 环境有 torch 和 transformers；numpy 装在用户目录，故用 -E -P 而不是 -I
/home/oc/miniconda3/bin/python -E -P $D/scripts/scene_graph/build_scene_graph.py $MAP \
  $D/scripts/scene_graph/configs/r2b_storage_ns.yaml /mnt/sda2/robot_download/r2b_nerfstudio/ns_data \
  $R/slam_data/models/grounding-dino-tiny $R/slam_data/models/chinese-clip-vit-large-patch14
# 约 40 秒（RTX 4090）。只要区域/地点/点位三层时加 --places-only，几秒，不需要 torch
```

地图包重新打包（`make_r2b_ns_map_package.py`）或点位库改动后，需要重跑构建。

## 调用

### Python（导航程序内）

```python
import sys
sys.path.insert(0, "/home/oc/projects/robot/slam_map_demo/scripts/scene_graph")
from scene_graph import SceneGraph

sg = SceneGraph.load("/home/oc/projects/robot/slam_map_demo/maps/r2b_storage_ns/2026-11-01.3")

# 1. 文字 → 候选节点（按匹配度排序）
sg.find("北侧通道里的纸箱")         # [{'node': {...obj_17...}, 'score': 0.864, 'match': '关键词+语义（CLIP）'}, ...]
sg.find("白色的小机器人", near=(x, y))  # near：同等匹配时优先离当前位置近的

# 2. 导航目标站位（给 A* / 行走控制）
g = sg.goal_pose("印着DGX STATION的纸箱")  # 也接受节点编号：obj_17 / wp_05 / region_north / place_009
# {'id': 'obj_17', 'name': '纸箱', 'type': 'object', 'x': 5.2, 'y': 1.6, 'yaw': 1.21,
#  'place': 'place_009', 'region': 'region_north', 'reachable': True}

# 3. 路线
sg.metric_path((x, y), "木箱")          # A* 折线 [[x, y], ...]，与 waypoint_tour.py 用的是同一个规划器
sg.topo_path((x, y), "木箱")            # 通行网络上的地点序列 ['place_000', ..., 'place_009']
sg.route_description((x, y), "木箱")    # ['主通道', '丁字路口', '北侧通道']

# 4. 图结构访问
sg.objects / sg.waypoints / sg.regions / sg.places    # 节点列表（字典，字段同 scene_graph.json）
sg.neighbors("obj_10", "next_to")       # [(相邻节点, 边), ...]
sg.objects_in("region_north", label="纸箱")
sg.region_at((x, y)); sg.nearest_place((x, y))
sg.grid                                 # nav_grid.NavGrid：碰撞高度场栅格、离障碍距离、A*
sg.to_networkx()                        # 需要 networkx

# 5. 给大语言模型的场景描述
print(sg.describe(position=(x, y)))
```

文字匹配规则：
- 查询里出现物体、区域、点位的名称或别名（配置文件中的 `aliases`）即“关键词命中”；同时出现区域名时只在该区域里找；
- 查询里还有颜色、文字等描述词（“白色的小机器人”“印着 DGX 的纸箱”），或同类物体有多个时，用中文 CLIP
  比较查询文字与物体图像向量来排序。检测器给的类别不一定对（白色小机器人被检测成手推车），
  语义排序会在全部物体里找，所以仍能找到；
- 没有任何关键词时，完全按 CLIP 语义检索。

导航进程所在的 Python 环境（`unitree-mujoco`）没有 torch。需要 CLIP 时，库会自动调用
`SCENE_GRAPH_TORCH_PYTHON`（默认 `/home/oc/miniconda3/bin/python`）在子进程里计算文字向量，每次约 10 秒；
纯关键词查询不受影响。频繁使用时，建议在有 torch 的环境里常驻一个进程。

### 命令行

```bash
cd /home/oc/projects/robot/slam_map_demo
P=/home/oc/miniconda3/envs/unitree-mujoco/bin/python
MAP=maps/r2b_storage_ns/2026-11-01.3
$P -I scripts/scene_graph/query_scene_graph.py $MAP find "蓝色手推车"
$P -I scripts/scene_graph/query_scene_graph.py $MAP goal "北侧通道里的纸箱" --from 0 0   # 站位、经过区域、A* 折线
$P -I scripts/scene_graph/query_scene_graph.py $MAP show obj_17                          # 节点全部字段和相连的边
$P -I scripts/scene_graph/query_scene_graph.py $MAP describe --from 0 0                  # 文字描述
$P -I scripts/scene_graph/query_scene_graph.py $MAP list object                          # 列出某类节点
# 加 --json 输出 JSON，便于其他语言的程序调用
```

### 让机器狗按文字走过去

```bash
# 仿真器没在运行时会自动在后台启动并弹出 MuJoCo 窗口（日志 /tmp/umj_slam_scene.log）；
# 有别的控制程序（如 patrol.sh 的巡航）在发指令时会提示先停掉它。到达最后一个目标后保持站立，Ctrl+C 退出，仿真器窗口保留
$P -I scripts/scene_graph/navigate_to.py dds $MAP "印着DGX STATION的纸箱" "蓝色手推车" "充电桩"
# 不开窗口跑一遍，输出轨迹图
$P -I scripts/scene_graph/navigate_to.py headless $MAP ../unitree_mujoco/unitree_robots/go2/scene_slam_r2b_storage_ns.xml \
  "北侧通道里的纸箱" "蓝色手推车" "白色的小机器人" "充电桩" --image output/navigate_scene_graph_test.png
```

## 两类导航算法如何使用

**A\* 等几何规划**：只需要 `goal_pose()` 把目标变成站位，再在 `sg.grid` 上规划（或直接用 `metric_path()`）。
场景图的通行网络只有十几个节点，也可以在上面先做一次图搜索（`topo_path()`），再对每一段做局部 A*，
地图大时比全图栅格 A* 快得多。

**VLN / 大语言模型规划**，常见两种接法：
1. 把 `describe()` 的文字和指令一起交给大语言模型，让它输出要去的节点编号序列（如 `["obj_17", "wp_01"]`），
   再逐个调用 `goal_pose()` + A* 执行。节点编号、区域和物体之间的关系都在描述里，模型不需要看图；
2. 指令中的名词短语（“印着 DGX 的纸箱”）直接用 `find()` 检索；方位关系（“北侧通道里的”）由区域限定处理；
   `route_description()` / `topo_path()` 给出分段子目标，便于逐段核对和重规划。

视觉语言模型需要把当前画面与地图对应时，可以用物体的 `object_features.npy`（CLIP 图像向量）
与当前画面的 CLIP 向量比较，判断看到的是哪个物体。

## 结果（2026-11-01）

- 构建：99 张左目照片，检测 413 次（另有 22 次与行人重叠被丢弃），合并为 26 个物体，用时约 40 秒；
- 文字查询：「充电桩」「北侧通道里的纸箱」「印着DGX STATION的纸箱」「白色的小机器人」「蓝色手推车」
  「贴着彩色色卡的东西」等均返回正确的物体或点位；
- 导航：无窗口仿真中从原点依次前往「北侧通道里的纸箱」「蓝色手推车」「白色的小机器人」「充电桩」，
  4/4 到达，用时 153 秒，位置误差 6~8 cm，最大倾斜 5.2°。

## 已知问题与限制

- 检测器的类别并不总对：白色小机器人被记为「手推车」（与「机器人」得分 0.502 比 0.497），
  贴在箱子上的二维码被记为「门」，叠着二维码的紫色周转箱被记为「纸箱」。按名称查询会受影响，按描述查询（走 CLIP）基本不受影响；
- 物体位置来自单次 10 秒采集，只有前向视角：货架背面、远处的物体包围盒偏大或偏小；
  离通道 2 m 以外的物体标记为走不到跟前（`approach.reachable = false`）；
- 物体之间的「在……上」关系是按包围盒判断的，相邻物体包围盒重叠时会误判；
- 采集时会动的物体（机器人、手推车）记录的是建图时的位置；
- 区域划分依赖配置文件里的种子点，换地图需要重新写 `configs/<地图>.yaml`；
- 场景图是静态的，运行中不随环境变化更新。
