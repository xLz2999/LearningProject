import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 Excel 文件
excel_file = pd.ExcelFile(r'C:\Users\86157\mycode\Groundwater depth forecast\Dataset.xlsx')

# 获取所有表名
sheet_name = excel_file.sheet_names
print(sheet_name)

# 遍历不同工作表表
for sheet in sheet_name:
    # 获取当前工作表的数据
    df = excel_file.parse(sheet)

    # 查看数据的基本信息
    print(f'sheet表名为{sheet}的基本信息：')
    df.info()

    # 查看数据集行数和列数
    rows, columns = df.shape
    print(f'sheet表名为{sheet}的数据集行数为{rows}，列数为{columns}')

    # 长表数据查看数据前几行信息（对齐版）
    print(f'sheet表名为{sheet}的前几行内容信息：')
    header = '\t'.join([col.ljust(20) for col in df.columns])
    print(header)
    for row in df.head().values:
        row_str = '\t'.join([str(val).ljust(20) for val in row])
        print(row_str)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置中文字体和字体大小
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.titlesize'] = 5  # 标题字体大小
plt.rcParams['axes.labelsize'] = 1  # 标签字体大小
plt.rcParams['xtick.labelsize'] = 3  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 3  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 3  # 图例字体大小

# 逐个显示图表
for i, sheet in enumerate(sheet_name):
    # 获取当前工作表的数据
    df = excel_file.parse(sheet)
    
    # 重置图像设置
    plt.close('all')
    
    # 创建第一个图表：时间序列趋势图
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Depth (m)'])
    plt.title(f'{sheet}站点埋深时间序列趋势图')
    plt.xlabel('日期')
    plt.ylabel('埋深（米）')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 创建第二个图表：相关性热力图
    plt.figure(figsize=(9, 7))
    corr = df[['Irrigation(万m³)', 'Rainfall(万m³)', 'Tem(℃)', 'Evaporation (万m³)', 'Depth (m)']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f'{sheet}站点各特征相关性热力图')
    plt.tight_layout()
    plt.show()
    
    # 创建第三个图表：箱线图
    plt.figure(figsize=(10, 6))
    df['Year'] = df['Date'].dt.year
    df.boxplot(column='Depth (m)', by='Year')
    plt.title(f'{sheet}站点不同年份平均埋深箱线图')
    plt.xlabel('年份')
    plt.ylabel('埋深（米）')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()