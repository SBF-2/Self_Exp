import pandas as pd
import matplotlib.pyplot as plt

# 加载CSV文件
file_path = 'code/loss_log.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 只选择 mode 列内容为 "eval" 的数据
eval_data = data[data['mode'] == 'eval']

# 提取需要绘制的损失列
loss_columns = [col for col in eval_data.columns if col.startswith('loss')]
steps = eval_data['global_step']

# 创建一个绘图
plt.figure(figsize=(16, 10))

# 为每个损失列绘制曲线
for loss in loss_columns[:2]:
	plt.plot(steps, eval_data[loss], label=loss)

# 添加标签和标题
plt.xlabel('Global Step')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.legend(title='Loss Types')

# 显示网格
plt.grid(True)

# 展示图形
plt.show()
