from openai import OpenAI
import json

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-5d58645e83ac4963ad925eefa7705e43"
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": "用户将提供给你一段对路况的描述，请你分析内容，并提取其中的关键信息，以 JSON 的形式输出。目前能够做的道路有：_make_merge，_make_straight，_make_double_u_turn（模拟 S 型弯），_make_intersection（最多出现一次）。需要给出每种路段的数量。其他需要的字段包括：vehicle_count（交通拥堵程度）、obstacle_count（障碍物数量）、obstacle_speed_range（障碍物速度范围）、double_u_turn_radius、other_vehicle_speed_range、road_sequence（值是 _make_xxx 的字符串列表，按实际顺序排列）。不能用‘高’‘快’等主观词汇描述，必须使用数字,速度30是指90km/h，other_vehicle_target_speed是指其他车辆的目标速度,应该是速度范围中的一个值,merge_obstacle_distance表示匝道汇入出障碍物的距离，正常为0.95，在（0,0.95）之间变动。所有 null 值用 0 表示。只允许输出以下字段：make_merge, make_straight, make_double_u_turn, make_intersection, vehicles_count, obstacle_count, obstacle_speed_range, double_u_turn_radius, other_vehicle_speed_range, road_sequence, straight_length，other_vehicle_target_speed,merge_obstacle_distance。所有字段的值必须为整数、浮点数，或形如 [最小值, 最大值] 的列表。输出 JSON。"
        },
        {
            "role": "user",
            "content": "乡村道路急转接200米直道（转弯半径60m），有倾倒的树木部分占用车道。一辆快车为避让树木向对向车道偏移，此时直道上有农用拖拉机牵引超宽耙具行驶，后方紧跟的校车因视线被耙具遮挡无法提前发现弯道险情。"
        }
    ]
)
print(completion.choices[0].message.content)

# 获取原始模型输出文本
json_text = completion.choices[0].message.content.strip()

# 自动清理 Markdown 代码块标记
if json_text.startswith("```json"):
    json_text = json_text[7:]  # 去掉 ```json\n
if json_text.endswith("```"):
    json_text = json_text[:-3]  # 去掉末尾的 ```

# 解析 JSON
try:
    parsed = json.loads(json_text)
except json.JSONDecodeError as e:
    print("解析失败，原始返回内容如下：")
    print(json_text)
    raise e

# 保存到 external_config.json
with open("D:/AUTO_Drive/external_config.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, ensure_ascii=False, indent=4)

print("✅ external_config.json 已保存成功！")

