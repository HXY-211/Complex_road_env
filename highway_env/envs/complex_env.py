from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import AbstractLane, StraightLane, CircularLane, LineType, SineLane
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.objects import Obstacle, Landmark
from highway_env import utils
from highway_env.utils import near_split
import numpy as np
import random
from highway_env.road.regulation import RegulatedRoad
import json
import random
import math

# 在 complex.py 中顶部导入配置 JSON 文件
try:
    with open("D:\AUTO_Drive\external_config.json", "r", encoding="utf-8") as f:
        EXTERNAL_CONFIG = json.load(f)
except FileNotFoundError:
    EXTERNAL_CONFIG = {}

class GoalRewardMixin:
    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict = None, p: float = 0.5
    ) -> float:
        """
        Compute the reward as a negative weighted distance including position and heading.
        """
        if info is None:
            info = {}

        ego = self.controlled_vehicles[0]
        weights = np.array(self.config.get("reward_weights", [1.0, 1.0, 0.1]))  # [x, y, heading]

        # 位置误差
        pos_error = np.abs(achieved_goal[:2] - desired_goal[:2])
        pos_dist = np.dot(pos_error, weights[:2])

        # 朝向误差（归一化到 [0, π]）
        heading_error = abs(ego.heading - ego.goal.heading) % (2 * np.pi)
        if heading_error > np.pi:
            heading_error = 2 * np.pi - heading_error

        heading_dist = weights[2] * heading_error

        total_error = pos_dist + heading_dist
        return -np.power(total_error, p)


    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        """
        Success if both position and heading are within thresholds.
        """
        ego = self.controlled_vehicles[0]

        # 位置误差
        pos_error = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        pos_thresh = self.config.get("position_threshold", 2.0)

        # 朝向误差（弧度）
        heading_error = abs(ego.heading - ego.goal.heading) % (2 * np.pi)
        if heading_error > np.pi:
            heading_error = 2 * np.pi - heading_error
        heading_thresh = np.deg2rad(self.config.get("heading_threshold_deg", 15))

        return pos_error < pos_thresh and heading_error < heading_thresh


    def goal_reward_active(self) -> bool:
        """Determine if the ego vehicle is in the designated goal (parking) zone segments."""
        ego_vehicle = self.controlled_vehicles[0]  # assume index 0 is the ego vehicle
        if not hasattr(self, "goal_segments"):
            return False  # if no goal zone defined, default to no goal reward
        # Check if the ego's current lane (road segment) is one of the goal zone segments
        return ego_vehicle.lane_index in self.goal_segments


class ComplexEnv(AbstractEnv,GoalRewardMixin):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {"type": "Kinematics"},
            "action": {"type": "ContinuousAction"},
            "controlled_vehicles": 1,
            "duration": 500,
            "screen_width": 1280,
            "screen_height": 720,
            "centering_position": [0.5, 0.5],               #"scaling": 1.5,

            # === 场景逻辑参数 ===
            "vehicles_count": 10,
            "parking_vehicles_count": 2,
            "dynamic_obstacle_count": 5,
            "spawn_probability": 0.6,
            "double_u_turn_radius": 50,
            "road_sequence": ["_make_merge", "_make_straight"],
            "randomize_road_sequence": False,
            "other_vehicle_target_speed":10,
            "merge_obstacle_distance": 0.95,

            # === Reward 配置 ===
            "distance_reward_weight": 0.5,
            "angle_reward_weight": 0.5,
            "success_goal_reward_weight": 1.0,
            "progress_reward_weight": 0.5,

            "high_speed_reward_weight": 0.2,
            "heading_reward_weight": 0.5,
            "lane_center_reward_weight": -0.4,
            "action_reward_weight": -0.01,
            "right_lane_reward_weight": 0.1,
            "forward_progress_weight": 0.9,
            "collision_reward_weight": -1.0,
            "out_of_road_reward_weight": -1.0,


            # === Goal 成功判定阈值 ===
            "position_threshold": 2.0,              # 距离误差判定成功 (米)
            "heading_threshold_deg": 15,            # 朝向误差判定成功 (度)
        })
        config.update(EXTERNAL_CONFIG)  # 自动合并来自文件的配置
        return config



    def _make_road(self):
        net = RoadNetwork()
        xx, yy = 0, 0
        id = 1

        # 初始化记录
        self.merge_ids = []
        self.straight_ids = []
        self.double_u_turn_ids = []
        self.parking_id = None
        self.intersection_id = None
        self.ego_lane_index = None
        self.has_intersection = False

        sequence = self.config["road_sequence"]
        if self.config.get("randomize_road_sequence", False):
            segment_pool = []
            for seg_name in ["_make_merge", "_make_straight", "_make_double_u_turn", "_make_intersection"]:
                count = self.config.get(seg_name.replace("_make_", "make_"), 0)
                segment_pool.extend([seg_name] * count)
            if "_make_intersection" in segment_pool:
                segment_pool.remove("_make_intersection")
                np.random.shuffle(segment_pool)
                mid = len(segment_pool) // 2
                segment_pool.insert(mid, "_make_intersection")
            else:
                np.random.shuffle(segment_pool)
            sequence = segment_pool

        # 记录当前段的起点 / 终点节点
        start_node = None
        end_node = None

        for segment in sequence:
            module = getattr(self, segment)

            # 标记当前路段 id
            if segment == "_make_merge":
                self.merge_ids.append(id)
            elif segment == "_make_straight":
                self.straight_ids.append(id)
            elif segment == "_make_double_u_turn":
                self.double_u_turn_ids.append(id)
            elif segment == "_make_intersection":
                self.intersection_id = id
                self.has_intersection = True

            # 如果当前不是首段，拼接短直道连接段
            if id != 1:
                net, xx, yy = self._add_short_straight_connector(net, xx, yy, start_node, f"a{id}")

            # 生成当前模块
            if segment == "_make_double_u_turn":
                net, xx, yy = module(net, xx, yy, id)
                end_node = f"g{id+1}"  # double_u_turn 结束在 g{id+1}
                id += 2
            elif segment == "_make_merge":
                net, xx, yy = module(net, xx, yy, id)
                end_node = f"d{id}"  # merge 结束在 d{id}
                id += 1
            elif segment == "_make_straight":
                net, xx, yy = module(net, xx, yy, id)
                end_node = f"b{id}"  # straight 结束在 b{id}
                id += 1
            elif segment == "_make_intersection":
                net, xx, yy = module(net, xx, yy, id)
                end_node = f"h{id}"  # intersection 结束在 h{id}
                id += 1
            else:
                raise ValueError(f"Unknown segment type: {segment}")



            start_node = end_node  # 更新起点

        # 停车段
        self.parking_id = id
        net = self._make_parking(net, xx, yy, id=id)

        self.road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        # 在 _make_road 末尾，安全选一个存在的车道作为 ego
        if self.ego_lane_index is None:
            for (from_node, to_node, lane_idx) in self.road.network.lanes_dict().keys():
                if lane_idx == 1:
                    self.ego_lane_index = (from_node, to_node, lane_idx)
                    break

        self._place_merge_obstacles()
        

    def _add_short_straight_connector(self, net, xx, yy, start_node, end_node):
        """
        在两个路段拼接处，添加 10m 的短直道连接段，双车道。
        start_node：上一个路段的终点
        end_node：下一个路段的起点
        """
        length = 10  # 连接段长度

        # 添加双车道短直道
        net.add_lane(
            start_node,
            end_node,
            StraightLane([xx, yy], [xx + length, yy], line_types=[LineType.NONE, LineType.CONTINUOUS_LINE]),
        )
        net.add_lane(
            start_node,
            end_node,
            StraightLane([xx, yy - StraightLane.DEFAULT_WIDTH],
                        [xx + length, yy - StraightLane.DEFAULT_WIDTH],
                        line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]),
        )

        xx += length  # 更新位置
        return net, xx, yy




    '''def _make_road(self):
        net = RoadNetwork()

        # Create road segments with unique IDs
        net, xx, yy = self._make_merge(net, xx=5, yy=6, id=1)
        net, xx, yy = self._make_straight(net, xx, yy, id=2)
        net, xx, yy = self._make_double_u_turn(net, xx, yy, id=3)
        net = self._make_parking(net, xx, yy, id=5)
        net, xx, yy = self._make_intersection(net, xx=5, yy=6)
        road = RegulatedRoad(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road = road'''

    def _place_merge_obstacles(self):
        for merge_id in self.merge_ids:
            try:
                # 匝道末端一般为 (bX, cX, 1)，因为 (bX, cX, 0) 是主路段
                lane = self.road.network.get_lane((f"b{merge_id}", f"c{merge_id}", 2))
                pos = lane.position(lane.length * self.config.get("merge_obstacle_distance", 0.95), 0)  # 不放在最边缘，稍微靠前一点
                obstacle = Obstacle(self.road, pos)
                obstacle.LENGTH = 4
                obstacle.WIDTH = 3
                self.road.objects.append(obstacle)
            except KeyError:
                continue


    def _make_merge(self, net, xx=0, yy=0, id=1):
        ends = [50, 80, 80, 50]  # Before, converging, merge, after
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        for i in range(2):
            net.add_lane(
                f"a{id}",
                f"b{id}",
                StraightLane([xx + 0, yy-StraightLane.DEFAULT_WIDTH + y[i]], [xx + sum(ends[:2]), yy-StraightLane.DEFAULT_WIDTH + y[i]], line_types=line_type[i]),
            )
            net.add_lane(
                f"b{id}",
                f"c{id}",
                StraightLane([xx + sum(ends[:2]), yy-StraightLane.DEFAULT_WIDTH + y[i]], [xx + sum(ends[:3]), yy-StraightLane.DEFAULT_WIDTH + y[i]], line_types=line_type_merge[i]),
            )
            net.add_lane(
                f"c{id}",
                f"d{id}",
                StraightLane([xx + sum(ends[:3]), yy-StraightLane.DEFAULT_WIDTH + y[i]], [xx + sum(ends), yy-StraightLane.DEFAULT_WIDTH + y[i]], line_types=line_type[i]),
            )

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([xx + 0, yy-StraightLane.DEFAULT_WIDTH + 6.5 + 4 + 4], [xx + ends[0], yy-StraightLane.DEFAULT_WIDTH + 6.5 + 4 + 4], line_types=[c, c], forbidden=True)
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0], line_types=[n, c], forbidden=True)
        net.add_lane(f"j{id}", f"k{id}", ljk)
        net.add_lane(f"k{id}", f"b{id}", lkb)
        net.add_lane(f"b{id}", f"c{id}", lbc)
        
        return net, xx + sum(ends), yy-StraightLane.DEFAULT_WIDTH + y[i]


    def _make_straight(self, net, xx=0, yy=0, id=2):
        length = self.config.get("straight_length", 100)
        net.add_lane(
            f"a{id}",
            f"b{id}",
            StraightLane([xx + 0, yy + 0], [xx + length, yy + 0], line_types=[LineType.NONE, LineType.CONTINUOUS_LINE]),
        )
        net.add_lane(
            f"a{id}",
            f"b{id}",
            StraightLane([xx, yy - SineLane.DEFAULT_WIDTH], [xx + length, yy - SineLane.DEFAULT_WIDTH], line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]),
        )
        return net, xx + length, yy


    def _make_u_turn_anticlockwise(self, net, xx, yy, id, radius):  # x, y means a点下面的坐标  
        # 逆时针从左往右-》从右往左
        length = 50

        offset = 2 * radius

        # Exit straight segment after the U-turn
        net.add_lane(  # cd下面道路
            f"c{id}", f"d{id}",
            StraightLane(
                [xx + length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH],
                [xx + 0, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )

        net.add_lane(  # cd上面道路
            f"c{id}", f"d{id}",
            StraightLane(
                [xx + length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset)],
                [xx + 0, yy - (2 * StraightLane.DEFAULT_WIDTH + offset)],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )

        # U-Turn lanes (counter-clockwise)
        center = [xx + length, yy - StraightLane.DEFAULT_WIDTH - offset // 2]
        radii = [radius, radius + StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                f"b{id}", f"c{id}",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90),
                    np.deg2rad(-90),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )

        # Entry straight segment before the U-turn
        net.add_lane(
            f"a{id}", f"b{id}",
            StraightLane(
                [xx + 0, yy - StraightLane.DEFAULT_WIDTH],
                [xx + length, yy - StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED),
            ),
        )
        net.add_lane(
            f"a{id}", f"b{id}",
            StraightLane(
                [xx + 0, yy],
                [xx + length, yy],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )

        return net, xx + 0, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH


    def _make_u_turn_clockwise(self, net, xx, yy, id, radius):  # x, y means a点下面的坐标  
        # 顺时针从右往左-》从左往右
        length = 50
        offset = 2 * radius

        # Exit straight segment after the U-turn
        net.add_lane(  # cd下面道路
            f"f{id}", f"g{id}",
            StraightLane(
                [xx - length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH],
                [xx + length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )

        net.add_lane(  # cd上面道路
            f"f{id}", f"g{id}",
            StraightLane(
                [xx - length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset)],
                [xx + length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset)],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            ),
        )

        # U-Turn lanes (clockwise)        
        center = [xx - length, yy - StraightLane.DEFAULT_WIDTH - offset // 2]
        radii = [radius, radius + StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[n, c], [c, s]]
        for lane in [0, 1]:
            net.add_lane(
                f"e{id}", f"f{id}",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90),
                    np.deg2rad(270),
                    clockwise=True,
                    line_types=line[lane],
                ),
            )

        # Entry straight segment before the U-turn
        net.add_lane(
            f"d{id-1}", f"e{id}",
            StraightLane(
                [xx + 0, yy - StraightLane.DEFAULT_WIDTH],
                [xx - length, yy - StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS_LINE),
            ),
        )
        net.add_lane(       # a下面道路
            f"d{id-1}", f"e{id}",
            StraightLane(
                [xx + 0, yy],
                [xx - length, yy],
                line_types=(LineType.CONTINUOUS_LINE, LineType.NONE),
            ),
        )

        return net, xx + length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH

    def _make_double_u_turn(self, net, xx, yy, id):  
        radius = self.config.get("double_u_turn_radius", 20)
        
        # 两次U-turn
        net, xx, yy = self._make_u_turn_anticlockwise(net, xx, yy, id, radius)
        id += 1
        net, xx, yy = self._make_u_turn_clockwise(net, xx, yy, id, radius)

        return net, xx, yy


    def _make_parking(self, net, xx, yy, id,spots: int = 14) -> None:
        """
        Create a road composed of straight adjacent lanes.

        :param spots: number of spots in the parking
        """
        width = 4.0
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = 10
        length = 8
        xx = xx + width * (spots // 2 + 1)
        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane(
                f"a{id}",
                f"b{id}",
                StraightLane([xx + x, yy + y_offset], [xx + x, yy + y_offset + length], width=width, line_types=lt),
            )
            net.add_lane(
                f"b{id}",
                f"c{id}",
                StraightLane([xx + x, yy - y_offset], [xx + x, yy - y_offset - length], width=width, line_types=lt),
            )
        return net

    def _make_intersection(self, net, xx, yy, id):
        """
        Add a right-turning entry (→ to ↑) followed by a 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns
        """

        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5
        left_turn_radius = right_turn_radius + lane_width
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50  # 50 in + 50 out

        # ===== 1. 添加入口段（向右走 → 向上拐） =====
        length = 30
        radius = 10
        offset = 2 * radius
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED

        # 直道 (→)
        net.add_lane(
            f"a{id}", f"b{id}",
            StraightLane(
                [xx + 0, yy - lane_width],
                [xx + length, yy - lane_width],
                line_types=(c, s),
            ),
        )
        net.add_lane(
            f"a{id}", f"b{id}",
            StraightLane(
                [xx + 0, yy],
                [xx + length, yy],
                line_types=(n, c),
            ),
        )

        # 圆弧 (逆时针 → ↑)
        center = [xx + length, yy - lane_width - radius]
        radii = [radius, radius + lane_width]
        line = [[c, s], [n, c]]
        for lane in [0, 1]:
            net.add_lane(
                f"b{id}", f"c{id}",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90),
                    np.deg2rad(0),
                    clockwise=False,
                    line_types=line[lane],
                ),
            )
        # 圆弧结束点坐标（圆弧出口位置）
        xx = xx + length + radius
        yy = yy - lane_width - radius

        # 加一段向上的直道（↑）
        straight_length = 30
        net.add_lane(
            f"c{id}", f"d{id}",
            StraightLane(
                [xx+lane_width, yy],
                [xx+lane_width, yy - straight_length],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE)
            ),
        )
        net.add_lane(
            f"c{id}", f"d{id}",
            StraightLane(
                [xx, yy],
                [xx, yy - straight_length],
                line_types=(LineType.CONTINUOUS_LINE, LineType.STRIPED)
            ),
        )
        xx=xx+(lane_width/2)  # 更新新位置，作为交叉口起点
        yy = yy - straight_length -access_length # 更新新位置，作为交叉口起点


        # ===== 2. 构建交叉口主结构（不动） =====
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane(
                "o" + str(corner),
                "ir" + str(corner),
                StraightLane(start + np.array([xx, yy]), end + np.array([xx, yy]), line_types=[s, c], priority=priority, speed_limit=10),
            )

            # Right turn
            r_center = rotation @ np.array([outer_distance, outer_distance]) + np.array([xx, yy])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner - 1) % 4),
                CircularLane(
                    r_center,
                    right_turn_radius,
                    angle + np.radians(180),
                    angle + np.radians(270),
                    line_types=[n, c],
                    priority=priority,
                    speed_limit=10,
                ),
            )

            # Left turn
            l_center = rotation @ (
                np.array([
                    -left_turn_radius + lane_width / 2,
                    left_turn_radius - lane_width / 2,
                ])
            ) + np.array([xx, yy])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 1) % 4),
                CircularLane(
                    l_center,
                    left_turn_radius,
                    angle + np.radians(0),
                    angle + np.radians(-90),
                    clockwise=False,
                    line_types=[n, n],
                    priority=priority - 1,
                    speed_limit=10,
                ),
            )

            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane(
                "ir" + str(corner),
                "il" + str((corner + 2) % 4),
                StraightLane(start + np.array([xx, yy]), end + np.array([xx, yy]), line_types=[s, n], priority=priority, speed_limit=10),
            )

            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane(
                "il" + str((corner - 1) % 4),
                "o" + str((corner - 1) % 4),
                StraightLane(end + np.array([xx, yy]), start + np.array([xx, yy]), line_types=[n, c], priority=priority, speed_limit=10),
            )


        length = 20
        radius = 40
        offset = 2 * radius
        angle = np.radians(90 * 1)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
        xx, yy = (start + np.array([xx, yy])).tolist()
        # Exit straight segment after the U-turn
        net.add_lane(  # cd下面道路
            f"g{id}", f"h{id}",
            StraightLane(
                [xx - length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH],
                [xx + 2*length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.NONE, LineType.CONTINUOUS_LINE),
            ),
        )

        net.add_lane(  # cd上面道路
            f"g{id}", f"h{id}",
            StraightLane(
                [xx - length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset)],
                [xx + 2*length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset)],
                line_types=(LineType.CONTINUOUS, LineType.STRIPED),
            ),
        )

        # U-Turn lanes (clockwise)        
        center = [xx - length, yy - StraightLane.DEFAULT_WIDTH - offset // 2]
        radii = [radius, radius + StraightLane.DEFAULT_WIDTH]
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        line = [[n, c], [c, s]]
        for lane in [0, 1]:
            net.add_lane(
                f"g{id}", f"h{id}",
                CircularLane(
                    center,
                    radii[lane],
                    np.deg2rad(90),
                    np.deg2rad(270),
                    clockwise=True,
                    line_types=line[lane],
                ),
            )

        # Entry straight segment before the U-turn
        net.add_lane(
            f"e{id}", f"f{id}",
            StraightLane(
                [xx + 0, yy - StraightLane.DEFAULT_WIDTH],
                [xx - length, yy - StraightLane.DEFAULT_WIDTH],
                line_types=(LineType.STRIPED, LineType.CONTINUOUS_LINE),
            ),
        )
        net.add_lane(       # a下面道路
            f"e{id}", f"f{id}",
            StraightLane(
                [xx + 0, yy],
                [xx - length, yy],
                line_types=(LineType.CONTINUOUS_LINE, LineType.NONE),
            ),
        )

        return net, xx + 2*length, yy - (2 * StraightLane.DEFAULT_WIDTH + offset) + StraightLane.DEFAULT_WIDTH



    
        road = self.road
        other_vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Ego vehicle
        if self.ego_lane_index is None:
            raise ValueError("Ego lane index was not properly initialized in _make_road.")
        ego_lane = road.network.get_lane(self.ego_lane_index)
        ego_vehicle = self.action_type.vehicle_class(
            road, ego_lane.position(30, 0), heading=ego_lane.heading_at(30), speed=30
        )
        ego_vehicle.color = (0, 255, 0)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.controlled_vehicles = [ego_vehicle]

        vehicles_count = self.config.get("vehicles_count", 1)
        speed_min, speed_max = self.config.get("other_vehicle_speed_range", [20, 30])

        # Merge vehicles: on-ramp + main road
        for merge_id in self.merge_ids:
            try:
                # 匝道车辆
                merging_lane = road.network.get_lane((f"j{merge_id}", f"k{merge_id}", 0))
                merging_vehicle = other_vehicle_type(
                    road,
                    merging_lane.position(10 + self.np_random.uniform(-10, 10), 0),
                    speed=20 + self.np_random.uniform(-2, 2)
                )
                merging_vehicle.target_speed = 30
                road.vehicles.append(merging_vehicle)

                # 匝道主路 other vehicles，覆盖两条车道，总数不超过 vehicles_count
                lane_indices = [(f"a{merge_id}", f"b{merge_id}", 0), (f"a{merge_id}", f"b{merge_id}", 1)]
                count = 0
                for lane_index in lane_indices:
                    if count >= vehicles_count:
                        break
                    lane = road.network.get_lane(lane_index)
                    for _ in range(vehicles_count):
                        if count >= vehicles_count:
                            break
                        position = lane.position(10 + self.np_random.uniform(-10, 80), 0)
                        speed = self.np_random.uniform(speed_min, speed_max)
                        vehicle = other_vehicle_type(road, position, speed=speed)
                        road.vehicles.append(vehicle)
                        count += 1
            except KeyError:
                continue  # 路段可能未生成成功，跳过'''

    '''def _make_vehicles(self):
        road = self.road
        other_vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Ego vehicle
        if self.ego_lane_index is None:
            raise ValueError("Ego lane index was not properly initialized in _make_road.")
        ego_lane = road.network.get_lane(self.ego_lane_index)
        ego_vehicle = self.action_type.vehicle_class(
            road, ego_lane.position(30, 0), heading=ego_lane.heading_at(30), speed=30
        )
        ego_vehicle.color = (0, 255, 0)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.controlled_vehicles = [ego_vehicle]

        vehicle_count = self.config.get("vehicles_count", 1)
        speed_min, speed_max = self.config.get("other_vehicle_speed_range", [20, 30])

        # 匝道 merging 车辆与主路 other vehicles
        for merge_id in self.merge_ids:
            try:
                # 匝道车辆
                merging_lane = road.network.get_lane((f"j{merge_id}", f"k{merge_id}", 0))
                merging_vehicle = other_vehicle_type(
                    road,
                    merging_lane.position(10 + self.np_random.uniform(-10, 10), 0),
                    speed=20 + self.np_random.uniform(-2, 2)
                )
                merging_vehicle.target_speed = 30
                road.vehicles.append(merging_vehicle)

                # 主路 other vehicles
                per_lane = vehicle_count // 2
                for lane_index in [(f"a{merge_id}", f"b{merge_id}", 0), (f"a{merge_id}", f"b{merge_id}", 1)]:
                    lane = road.network.get_lane(lane_index)
                    positions = []
                    for _ in range(per_lane):
                        for _ in range(10):  # 最多尝试10次找合适位置
                            pos_val = self.np_random.uniform(10, 80)
                            if all(abs(pos_val - p) > 8 for p in positions):
                                positions.append(pos_val)
                                break
                    for pos_val in positions:
                        pos = lane.position(pos_val, 0)
                        speed = self.np_random.uniform(speed_min, speed_max)
                        vehicle = other_vehicle_type(road, pos, speed=speed)
                        road.vehicles.append(vehicle)
            except KeyError:
                continue

        # 直道 straight 路段 other vehicles
        for straight_id in self.straight_ids:
            try:
                per_lane = vehicle_count // 2
                for lane_index in [(f"a{straight_id}", f"b{straight_id}", 0), (f"a{straight_id}", f"b{straight_id}", 1)]:
                    lane = road.network.get_lane(lane_index)
                    positions = []
                    for _ in range(per_lane):
                        for _ in range(10):
                            pos_val = self.np_random.uniform(10, 80)
                            if all(abs(pos_val - p) > 8 for p in positions):
                                positions.append(pos_val)
                                break
                    for pos_val in positions:
                        pos = lane.position(pos_val, 0)
                        speed = self.np_random.uniform(speed_min, speed_max)
                        vehicle = other_vehicle_type(road, pos, speed=speed)
                        road.vehicles.append(vehicle)
            except KeyError:
                continue'''

    def _make_vehicles(self):
        road = self.road
        other_vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        # Ego vehicle
        if self.ego_lane_index is None:
            raise ValueError("Ego lane index was not properly initialized in _make_road.")
        ego_lane = road.network.get_lane(self.ego_lane_index)
        ego_vehicle = self.action_type.vehicle_class(
            road, ego_lane.position(10, 0), speed=30
        )
        ego_vehicle.color = (0, 255, 0)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle
        self.controlled_vehicles = [ego_vehicle]

        # Initial batch creation
        self._generate_merge_vehicles()
        self._generate_straight_vehicles()
        self._generate_double_u_turn_vehicles()
        self._generate_intersection_vehicles()
        self._generate_parking_goal_and_vehicles()
        self._generate_dynamic_obstacles()


    '''def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1

        self._remove_finished_other_vehicles()
        self._clear_vehicles()
        self.remove_invalid_other_vehicles()
        # 每隔 N 步再生成车辆
        if self._step_count % 5 == 0:  # 控制周期：每x步执行一次生成
            self._generate_merge_vehicles()
            self._generate_double_u_turn_vehicles()
            self._generate_intersection_vehicles()
        if self._step_count % 12 == 0:  # 控制周期：每x步执行一次生成
            self._generate_straight_vehicles()
        print("ego vehicle position:", self.controlled_vehicles[0].position)
        ego= self.controlled_vehicles[0]
        achieved_goal = np.array(ego.position)            # current position of ego vehicle
        desired_goal = np.array(ego.goal.position)        # target goal position (parking spot)
        print("reward:",self.compute_reward(achieved_goal, desired_goal))

        return obs, reward, terminated, truncated, info'''

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self._step_count += 1

        # 清除无效车辆
        self._remove_finished_other_vehicles()
        self._clear_vehicles()
        self.remove_invalid_other_vehicles()

        # 定期生成新车辆
        if self._step_count % 5 == 0:
            self._generate_merge_vehicles()
            self._generate_double_u_turn_vehicles()
            self._generate_intersection_vehicles()
        if self._step_count % 12 == 0:
            self._generate_straight_vehicles()
            # self._generate_dynamic_obstacles()  # 如果需要动态障碍物生成可打开

        # 打印调试信息（可选）
        ego = self.controlled_vehicles[0]
        '''print("ego vehicle position:", ego.position)
        if hasattr(ego, "goal") and ego.goal:
            achieved_goal = np.array(ego.position)
            desired_goal = np.array(ego.goal.position)
            print("reward:", self.compute_reward(achieved_goal, desired_goal))'''

        # ===== intersection 路段逻辑 =====
        # 仅在首次进入 intersection 执行一次（route_planned 标记）
        if self.has_intersection and not getattr(self, "route_planned", False):
            if ego.lane_index is not None:
                origin, destination, lane_idx = ego.lane_index

                # 进入 intersection 的核心段，触发规划
                if origin.startswith("d") and destination.startswith("e") and origin[1:] == destination[1:]:
                    '''print("[Intersection] 已进入交叉口，开始规划前往 o1")'''
                    ego.plan_route_to("o1")
                    self.route_planned = True

        return obs, reward, terminated, truncated, info



    def _generate_merge_vehicles(self):
        road = self.road
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_count = self.config.get("vehicles_count", 1)
        speed_min, speed_max = self.config.get("other_vehicle_speed_range", [20, 25])

        count = 0
        for merge_id in self.merge_ids:
            try:
                # 匝道车辆，仅在当前匝道没有车辆时生成
                merging_lane = road.network.get_lane((f"j{merge_id}", f"k{merge_id}", 0))
                if not any(v.lane_index == (f"j{merge_id}", f"k{merge_id}", 0) for v in road.vehicles):
                    merging_vehicle = vehicle_type(
                        road,
                        merging_lane.position(10 + self.np_random.uniform(-5, 5), 0),
                        speed=20 + self.np_random.uniform(-2, 2)
                    )
                    merging_vehicle.target_speed = 30
                    merging_vehicle.plan_route_to(f"d{merge_id}")
                    road.vehicles.append(merging_vehicle)

                # 主路车辆，控制总量不超过 vehicles_count
                lane_list = [(f"a{merge_id}", f"b{merge_id}", 0), (f"a{merge_id}", f"b{merge_id}", 1)]
                per_lane = vehicle_count // 2

                for lane_index in lane_list:
                    if count >= vehicle_count:
                        break
                    lane = road.network.get_lane(lane_index)
                    positions = []
                    for _ in range(per_lane):
                        if count >= vehicle_count:
                            break
                        # 随机位置，避免太近
                        for _ in range(10):
                            pos_val = self.np_random.uniform(15, 60)
                            if all(abs(pos_val - p) > 8 for p in positions):
                                positions.append(pos_val)
                                break
                    for pos_val in positions:
                        pos = lane.position(pos_val, 0)
                        speed = self.np_random.uniform(speed_min, speed_max)
                        vehicle = vehicle_type(road, pos, speed=speed)
                        vehicle.target_speed = self.config.get("other_vehicle_target_speed", 20)
                        vehicle.plan_route_to(f"d{merge_id}")
                        road.vehicles.append(vehicle)
                        count += 1

            except KeyError:
                continue

    def _generate_straight_vehicles(self):
        road = self.road
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_count = self.config.get("vehicles_count", 1)
        speed_min, speed_max = self.config.get("other_vehicle_speed_range", [15, 25])
        length = self.config.get("straight_length", 100)

        count = 0
        for sid in self.straight_ids:
            try:
                lane_list = [(f"a{sid}", f"b{sid}", 0), (f"a{sid}", f"b{sid}", 1)]
                per_lane = vehicle_count // 2

                for lane_index in lane_list:
                    if count >= vehicle_count:
                        break
                    lane = road.network.get_lane(lane_index)
                    positions = []

                    for _ in range(per_lane):
                        if count >= vehicle_count:
                            break
                        for _ in range(10):  # 尝试多次找不冲突的位置
                            pos_val = self.np_random.uniform(10, length / 2)
                            if all(abs(pos_val - p) > 8 for p in positions):
                                positions.append(pos_val)
                                break

                    for pos_val in positions:
                        pos = lane.position(pos_val, 0)
                        speed = self.np_random.uniform(speed_min, speed_max)
                        vehicle = vehicle_type(road, pos, speed=speed)
                        vehicle.target_speed = self.config.get("other_vehicle_target_speed", 20)
                        road.vehicles.append(vehicle)
                        count += 1

            except KeyError:
                continue


    def _generate_double_u_turn_vehicles(self):
        road = self.road
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_count = self.config.get("vehicles_count", 1)
        speed_min, speed_max = self.config.get("other_vehicle_speed_range", [15, 25])

        count = 0
        for duid in self.double_u_turn_ids:
            try:
                if count >= vehicle_count:
                    break
                target = f"g{duid + 1}"
                lane_list = [(f"a{duid}", f"b{duid}", 0), (f"a{duid}", f"b{duid}", 1)]
                per_lane = vehicle_count // 2

                for lane_index in lane_list:
                    if count >= vehicle_count:
                        break
                    lane = road.network.get_lane(lane_index)
                    positions = []

                    for _ in range(per_lane):
                        if count >= vehicle_count:
                            break
                        for _ in range(10):
                            pos_val = self.np_random.uniform(15, 40)
                            if all(abs(pos_val - p) > 8 for p in positions):
                                positions.append(pos_val)
                                break

                    for pos_val in positions:
                        pos = lane.position(pos_val, 0)
                        speed = self.np_random.uniform(speed_min, speed_max)
                        vehicle = vehicle_type(road, pos, speed=speed)
                        vehicle.target_speed = self.config.get("other_vehicle_target_speed", 20)
                        vehicle.plan_route_to(target)
                        vehicle.randomize_behavior()
                        road.vehicles.append(vehicle)
                        count += 1

            except KeyError:
                continue


    def _generate_intersection_vehicles(self):
        if not getattr(self, "has_intersection", False):
            return  # 如果没有 intersection，不生成车辆
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        n_vehicles = self.config.get("other_vehicles_count", 1)  # 适当减少频次避免阻塞

        for _ in range(n_vehicles):
            self._spawn_vehicle(longitudinal=self.np_random.uniform(0, 30), spawn_probability=0.6)

    def _spawn_vehicle(
        self,
        longitudinal: float = 0,
        position_deviation: float = 1.0,
        speed_deviation: float = 1.0,
        spawn_probability: float = 0.6,
        go_straight: bool = False,
    ) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        lane_index = ("o" + str(route[0]), "ir" + str(route[0]), 0)

        # 确保 lane 存在
        try:
            _ = self.road.network.get_lane(lane_index)
        except KeyError:
            return  # 如果没有 intersection 路段就跳过

        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(
            self.road,
            lane_index,
            longitudinal=(longitudinal + 5 + self.np_random.normal() * position_deviation),
            speed=8 + self.np_random.normal() * speed_deviation,
        )

        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle



    def _remove_finished_other_vehicles(self):
        """
        Remove other (non-ego) vehicles that have exited the end of their respective road segments.
        Works for merge (c{id}->d{id}), straight (a{id}->b{id}), and double_u_turn (c{id+1}->d{id+1}).
        """

        def is_leaving(vehicle):
            if not hasattr(vehicle, "lane_index") or vehicle.lane_index is None:
                return False
            from_node, to_node, _ = vehicle.lane_index

            # Merge 路段末尾
            for mid in self.merge_ids:
                if from_node == f"c{mid}" and to_node == f"d{mid}":
                    try:
                        return vehicle.lane.local_coordinates(vehicle.position)[0] >= vehicle.lane.length - 4 * vehicle.LENGTH
                    except Exception:
                        return False

            # Straight 路段末尾
            for sid in self.straight_ids:
                if from_node == f"a{sid}" and to_node == f"b{sid}":
                    try:
                        return vehicle.lane.local_coordinates(vehicle.position)[0] >= vehicle.lane.length - 4 * vehicle.LENGTH
                    except Exception:
                        return False

            # Double U-Turn 的第二段出口
            for duid in self.double_u_turn_ids:
                actual_id = duid + 1  # 注意 double 的第二段 ID 是 +1 后生成的
                if from_node == f"f{actual_id}" and to_node == f"g{actual_id}":
                    try:
                        return vehicle.lane.local_coordinates(vehicle.position)[0] >= vehicle.lane.length - 4 * vehicle.LENGTH
                    except Exception:
                        return False

            return False

        self.road.vehicles = [
            v for v in self.road.vehicles
            if v in self.controlled_vehicles or not is_leaving(v)
        ]


    def _clear_vehicles(self) -> None:
        def is_intersection_leaving(vehicle):
            if not hasattr(vehicle, "lane_index") or vehicle.lane_index is None:
                return False
            from_node, to_node, _ = vehicle.lane_index
            return (
                from_node.startswith("il") and to_node.startswith("o")
                and vehicle.lane.local_coordinates(vehicle.position)[0]
                >= vehicle.lane.length - 4 * vehicle.LENGTH
            )

        self.road.vehicles = [
            v for v in self.road.vehicles
            if v in self.controlled_vehicles or not is_intersection_leaving(v)
        ]

    def _generate_parking_goal_and_vehicles(self):
        """
        Generate parking goals and place other static vehicles.
        Only affects the final parking segment, defined by self.parking_id.
        """
        

        road = self.road
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        parking_id = self.parking_id
        vehicles_count = self.config.get("parking_vehicles_count", 1)

        # 找到所有可用的停车车道索引
        parking_lanes = [
            (f"a{parking_id}", f"b{parking_id}", i) for i in range(14)
        ] + [
            (f"b{parking_id}", f"c{parking_id}", i) for i in range(14)
        ]

        # 复制一份用于分配
        empty_spots = parking_lanes.copy()

        # 给每辆 ego 分配 goal
        for vehicle in self.controlled_vehicles:
            if not empty_spots:
                break
            lane_index = random.choice(empty_spots)
            lane = road.network.get_lane(lane_index)
            goal = Landmark(road, lane.position(lane.length / 2, 0), heading=lane.heading)
            vehicle.goal = goal
            vehicle.goal.color = (0, 225, 0)
            road.objects.append(goal)
            empty_spots.remove(lane_index)

        # 放置其他车辆占位
        for _ in range(vehicles_count):
            if not empty_spots:
                break
            lane_index = random.choice(empty_spots)
            lane = road.network.get_lane(lane_index)
            vehicle = vehicle_type.make_on_lane(road, lane_index, longitudinal=4, speed=0)
            road.vehicles.append(vehicle)
            empty_spots.remove(lane_index)

    def _generate_dynamic_obstacles(self):
        road = self.road
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        obstacle_count = self.config.get("obstacle_count", 2)
        speed_min, speed_max = self.config.get("obstacle_speed_range", [0, 5])

        def generate_lane_candidates():
            lane_candidates = []
            # Merge 路段: a→b, b→c, c→d
            for mid in self.merge_ids:
                for seg_pair in [("a", "b"), ("b", "c"), ("c", "d")]:
                    for i in range(2):
                        lane_candidates.append((f"{seg_pair[0]}{mid}", f"{seg_pair[1]}{mid}", i))

            # Straight 路段: a→b
            for sid in self.straight_ids:
                for i in range(2):
                    lane_candidates.append((f"a{sid}", f"b{sid}", i))

            # S弯 double U-turn: a→b→c→d→e→f→g (注意后半段 ID+1)
            for duid in self.double_u_turn_ids:
                next_id = duid + 1
                segs = [("a", "b", duid), ("b", "c", duid), ("c", "d", duid),
                        ("d", "e", next_id), ("e", "f", next_id), ("f", "g", next_id)]
                for seg in segs:
                    for i in range(2):
                        lane_candidates.append((f"{seg[0]}{seg[2]}", f"{seg[1]}{seg[2]}", i))
            return lane_candidates

        lane_candidates = generate_lane_candidates()
        self.np_random.shuffle(lane_candidates)
        lane_candidates = lane_candidates[:obstacle_count]

        for lane_index in lane_candidates:
            try:
                lane = road.network.get_lane(lane_index)
                longitudinal = self.np_random.uniform(0, lane.length - 5)
                speed = self.np_random.uniform(speed_min, speed_max)
                obstacle = vehicle_type(road, lane.position(longitudinal, 0), speed=speed)

                # 设置随机朝向与尺寸
                obstacle.heading = self.np_random.uniform(-np.pi, np.pi)
                obstacle.LENGTH = 3  # 比常规车辆短
                obstacle.WIDTH = 3
                obstacle.color = (255, 255, 0)

                road.vehicles.append(obstacle)
            except KeyError:
                continue

    def remove_invalid_other_vehicles(self):
        """
        清除普通 other vehicles（非 ego、非 dynamic obstacle），如果：
        - 与其他车辆发生碰撞；
        - 已驶出道路；
        - 初始位置与 ego 过近；
        dynamic obstacle 不清除（通过颜色判断）。
        """
        min_distance = 10.0
        ego_vehicle = self.vehicle
        valid_vehicles = []

        for v in self.road.vehicles:
            # 保留 ego
            if v is ego_vehicle:
                valid_vehicles.append(v)
                continue

            # 保留动态障碍物（黄色标记）
            if hasattr(v, "color") and v.color == (255, 255, 0):
                valid_vehicles.append(v)
                continue

            # 初始位置与 ego 太近
            if np.linalg.norm(v.position - ego_vehicle.position) < min_distance:
                continue

            # 是否驶出道路
            if hasattr(v, "on_road") and not v.on_road:
                continue

            # 是否发生碰撞
            if hasattr(v, "crashed") and v.crashed:
                continue

            # 其余合法 other vehicles保留
            valid_vehicles.append(v)

        self.road.vehicles = valid_vehicles



    def _reset(self):
        self._step_count = 0
        self._make_road()
        self._make_vehicles()
        self.route_planned = False  # 每次重置都重新设置
        self.reverse_steps = 0
        self.stagnation_steps = 0

    def _rewards(self, action):
        ego = self.controlled_vehicles[0] if hasattr(self, "controlled_vehicles") else self.vehicle
        rewards = {}
    
        # ===== 停车阶段 =====
        if self.goal_reward_active():
            goal_pos = getattr(self, "goal_position", np.zeros(2))
            goal_heading = getattr(self, "goal_heading", 0.0)
            ego_pos = np.array(ego.position[:2])
            dist_error = np.linalg.norm(ego_pos - goal_pos)
            heading_diff = (ego.heading - goal_heading + np.pi) % (2 * np.pi) - np.pi
            orientation_error = abs(heading_diff)
    
            # 归一化距离和角度奖励
            dist_reward = max(0.0, 1.0 - dist_error / max(self.config.get("distance_threshold", 2.0), 1e-6))
            angle_reward = max(0.0, 1.0 - orientation_error / max(self.config.get("angle_threshold", 0.1), 1e-6))
    
            rewards["distance_reward"] = dist_reward
            rewards["angle_reward"] = angle_reward
            rewards["success_goal_reward"] = 1.0 if (dist_error < self.config.get("distance_threshold", 2.0)
                                                      and orientation_error < self.config.get("angle_threshold", 0.1)) else 0.0
    
            # 进度奖励（归一化到 [0,1]）
            if hasattr(self, "prev_dist_to_goal"):
                progress = self.prev_dist_to_goal - dist_error
                rewards["progress_reward"] = max(0.0, min(progress / 5.0, 1.0))  # 正向小进度奖励，归一化
            else:
                rewards["progress_reward"] = 0.0
            self.prev_dist_to_goal = dist_error
    
        # ===== 行驶阶段 =====
        else:
            # 速度奖励 (0 ~ 1)
            vx, vy = ego.velocity if hasattr(ego, "velocity") else (0.0, 0.0)
            heading = ego.heading
            v_forward = vx * np.cos(heading) + vy * np.sin(heading)
            v_min, v_max = self.config.get("reward_speed_range", [0, 30])
            speed_reward = (min(max(v_forward, v_min), v_max) - v_min) / (v_max - v_min) if v_max > v_min else 0.0
    
            # 朝向奖励 (0 ~ 1)
            heading_reward = 0.0
            if hasattr(ego.lane, "heading_at"):
                lane_dir = ego.lane.heading_at(ego.lane.local_coordinates(ego.position)[0])
                heading_diff = abs((heading - lane_dir + np.pi) % (2 * np.pi) - np.pi)
                heading_reward = (1.0 + np.cos(heading_diff)) / 2.0  # 0.0(逆行) ~ 1.0(同向)
    
            # 车道居中奖励 (0 ~ 1)，越偏离中心越小
            if ego.lane:
                lateral_offset = ego.lane.local_coordinates(ego.position)[1]
                max_offset = ego.lane.width / 2
                lane_center_reward = max(0.0, 1.0 - min(abs(lateral_offset) / max_offset, 1.0))  # 中心1.0，边缘0.0
            else:
                lane_center_reward = 0.0
    
            # 动作惩罚 (0 ~ 1)：幅度越小越好
            act = np.array(action, dtype=float).ravel()
            max_norm = np.sqrt(len(act))
            action_penalty = 1.0 - min(np.linalg.norm(act) / max_norm, 1.0) if max_norm > 0 else 1.0  # 1.0最好，0最差
    
            # intersection靠右奖励 (0或0.1)
            right_lane_reward = 0.0
            if hasattr(self, "intersection_id") and ego.lane_index is not None:
                start_lane, end_lane, lane_idx = ego.lane_index
                inter_id = str(self.intersection_id)
                if str(start_lane) == f"a{inter_id}" and str(end_lane) == f"b{inter_id}" and lane_idx == 0:
                    right_lane_reward = 0.1
    
            # 进度奖励 (0 ~ 1)
            if hasattr(self, "prev_pos"):
                long_prev = ego.lane.local_coordinates(self.prev_pos)[0] if ego.lane else 0.0
                long_now = ego.lane.local_coordinates(ego.position)[0] if ego.lane else 0.0
                forward_progress = max(0.0, min((long_now - long_prev) / 5.0, 1.0))
            else:
                forward_progress = 0.0
            self.prev_pos = ego.position.copy()
    
            # 出界 / 碰撞惩罚 (0 ~ -1)
            collision_reward = -1.0 if ego.crashed else 0.0
            out_of_road_reward = -1.0 if not ego.on_road else 0.0
    
            # 统一写入
            rewards.update({
                "high_speed_reward": speed_reward,
                "heading_reward": heading_reward,
                "lane_center_reward": lane_center_reward,
                "action_reward": action_penalty,
                "right_lane_reward": right_lane_reward,
                "forward_progress": forward_progress,
                "collision_reward": collision_reward,
                "out_of_road_reward": out_of_road_reward
            })
    
        return rewards



    def _reward(self, action):
        """
        根据当前 action，内部调用 _rewards()，聚合计算最终奖励。
        返回一个标量 float 供 sb3 使用。
        """
        # === 先拿到每个子奖励 ===
        rewards = self._rewards(action)

        # === 行驶阶段 or 停车阶段 ===
        if self.goal_reward_active():
            # ------ 停车阶段奖励计算 ------
            total = (
                self.config.get("distance_reward_weight", 0.5) * rewards.get("distance_reward", 0.0)
                + self.config.get("angle_reward_weight", 0.5) * rewards.get("angle_reward", 0.0)
                + self.config.get("success_goal_reward_weight", 1.0) * rewards.get("success_goal_reward", 0.0)
                + self.config.get("progress_reward_weight", 0.3) * rewards.get("progress_reward", 0.0)
            )
            # 归一化到 [0, 1]
            max_total = self.config.get("success_goal_reward", 1.0)
            reward = utils.lmap(total, [0.0, max_total], [0.0, 1.0])

        else:
            # ------ 行驶阶段奖励计算 ------
            total = sum(
                self.config.get(name + "_weight", 0.0) * rewards.get(name, 0.0)
                for name in [
                    "high_speed_reward",
                    "heading_reward",
                    "lane_center_reward",
                    "action_reward",
                    "right_lane_reward",
                    "forward_progress",
                    "collision_reward",
                    "out_of_road_reward"
                ]
            )
            # 归一化：行驶阶段最小值 = 碰撞惩罚+动作惩罚，最大值 = 高速 + right_lane
            max_total = (
                self.config.get("high_speed_reward_weight", 0.4)
                + self.config.get("heading_reward_weight", 0.5)
                + self.config.get("right_lane_reward_weight", 0.1)
                + self.config.get("forward_progress_weight", 0.9)
            )
            min_total = (
                self.config.get("collision_reward_weight", -1.0)
                + self.config.get("out_of_road_reward_weight", -1.0)
                + min(0.0, self.config.get("action_reward_weight", -0.01))
                + min(0.0, self.config.get("lane_center_reward_weight", -0.2))
            )
            reward = utils.lmap(total, [min_total, max_total], [0.0, 1.0])

            #print(f"Step: reward={reward}, total={total}, min={min_total}, max={max_total}")

        return float(np.clip(reward, 0.0, 1.0))



    def _is_terminated(self):
        ego = self.controlled_vehicles[0] if hasattr(self, "controlled_vehicles") else self.vehicle
        # 成功泊车
        '''if self.goal_reward_active():
            goal_pos = getattr(self, "goal_position", np.zeros(2))
            goal_heading = getattr(self, "goal_heading", 0.0)
            dist_error = np.linalg.norm(np.array(ego.position) - goal_pos)
            heading_diff = abs((ego.heading - goal_heading + np.pi) % (2 * np.pi) - np.pi)
            if dist_error < 0.5 and heading_diff < math.radians(5):
                return True'''
        # 撞车 / 出界
        if ego.crashed or not ego.on_road:
            return True
        # 长时间停滞
        if ego.speed < 0.1:
            self.stagnation_steps += 1
        else:
            self.stagnation_steps = 0
        if self.stagnation_steps > 50:
            return True
        # 逆行检测
        if hasattr(ego.lane, "heading_at"):
            lane_dir = ego.lane.heading_at(ego.lane.local_coordinates(ego.position)[0])
            heading_diff = abs((ego.heading - lane_dir + np.pi) % (2 * np.pi) - np.pi)
            if heading_diff > math.radians(135):
                self.reverse_steps += 1
            else:
                self.reverse_steps = 0
            if self.reverse_steps > 20:
                return True
        return False



    def _is_truncated(self) -> bool:
        """Check time-based truncation (if episode exceeded max duration)."""
        # (Assuming ComplexEnv uses a time step counter like self.time and config "duration")
        return getattr(self, "time", 0) >= self.config.get("duration", 500)