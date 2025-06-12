from openai import OpenAI
import json

client = OpenAI(
    base_url="https://api.deepseek.com/",
    api_key="sk-352efd0e050d43baa56b725c38a0f007"
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": "在交通的场景下，能根据这些例子，再生成一些危险的例子吗，我的道路有弯道，直道，匝道汇入（匝道尽头的障碍物位置可以改变，匝道汇入的车辆正常行驶），十字路口，停车场，只涉及车辆，随机出现的物体（可以移动也可以固定,出现位置随机），没有红绿灯，不考虑逆行，单向双车道："
        },
        {
            "role": "user",
            "content": "Novel Scenario：Pattern that was not observed during the training process, but does not increase the potential for collision. Truck appears from a side road (but is going to stop). Accessing the freeway"
        }
    ]
)
print(completion.choices[0].message.content)

'''# 获取原始模型输出文本
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
with open("D:/AUTO_Drive/description.json", "w", encoding="utf-8") as f:
    json.dump(parsed, f, ensure_ascii=False, indent=4)

print("✅ external_config.json 已保存成功！")'''

