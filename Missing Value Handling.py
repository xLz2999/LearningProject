import pandas as pd

# 读取 Excel 文件
excel_file = pd.ExcelFile(r'C:\Users\86157\mycode\Groundwater depth forecast\Dataset.xlsx')

# 获取所有表名
sheet_name = excel_file.sheet_names
print(sheet_name)

# 初始化 Excel 写入器
with pd.ExcelWriter(r'C:\Users\86157\mycode\Groundwater depth forecast\Dataset_processed.xlsx') as writer:
    # 遍历不同工作表
    for sheet in sheet_name:
        # 获取当前工作表的数据
        df = excel_file.parse(sheet)
        
        # 确保 Date 列是 datetime 类型
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 设置日期索引
        df.set_index('Date', inplace=True)
        
        # 对存在缺失值的列进行时间插值
        columns = ['Irrigation(万m³)', 'Depth (m)']
        df[columns] = df[columns].interpolate(method='time')
        
        # 将处理后的数据写入新的 Excel 文件
        df.to_excel(writer, sheet_name=sheet)
        
print("数据处理完成，已保存至 Dataset_processed.xlsx")