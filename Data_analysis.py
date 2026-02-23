
# 新人用AI写代码，一方面是学习代码编程，一方面也在学习数据分析流程
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 全局设置（解决中文乱码、图表风格）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")  # 网格风格，让图表更清晰



# 加载数据集（路径按你的实际位置调整，这里用/mnt/air_pollution.csv）
df = pd.read_csv("air_pollution.csv", encoding="utf-8")

# 查看数据前5行（验证表头是否匹配：city、country、2017-2023）
print("=== 数据前5行 ===")
print(df.head())

# 查看数据基础信息（列名、数据类型、非空值数量）
print("\n=== 数据基础信息 ===")
print(df.info())

# 统计关键维度：有多少个城市、多少个国家（空间范围）
city_count = df["city"].nunique()  # 唯一城市数量
country_count = df["country"].nunique()  # 唯一国家数量
print(f"\n=== 数据覆盖范围 ===")
print(f"总记录数：{len(df)} 条")
print(f"覆盖城市数：{city_count} 个")
print(f"覆盖国家数：{country_count} 个")
print(f"覆盖国家列表：{df['country'].unique()}")  # 查看具体国家

# 查看PM2.5数值统计（2017-2023列）：均值、极值、分位数（判断异常值）
pm_cols = [str(year) for year in range(2017, 2024)]  # 生成2017-2023列名列表
print("\n=== 2017-2023年PM2.5数值统计（μg/m³）===")
print(df[pm_cols].describe().round(2))  # 保留2位小数，更易读

# 统计各列缺失值（重点看年度PM2.5列是否有缺失）
print("\n=== 各列缺失值比例（%）===")
missing_rate = (df.isnull().sum() / len(df) * 100).round(2)
print(missing_rate[missing_rate > 0])  # 只显示有缺失的列（无缺失则不输出）










# 1. 分类列转换（city、country是字符串，转为category类型减少内存）
df["city"] = df["city"].astype("category")
df["country"] = df["country"].astype("category")

# 2. 数值列验证（2017-2023列必须是数值类型，若为字符串则转换）
for col in pm_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # errors="coerce"：无法转换的值设为NaN

print("=== 清洗后数据类型 ===")
print(df.dtypes)











# 策略：年度PM2.5缺失→用该城市“相邻年份均值”填充（比全局均值更贴合城市实际）
for col in pm_cols:
    # 对每一年的缺失值，用该城市前一年和后一年的均值填充（若存在）
    for idx in df[df[col].isnull()].index:
        city = df.loc[idx, "city"]  # 当前缺失值对应的城市
        year = int(col)  # 当前年份
        # 找相邻年份（如2018缺失，找2017和2019）
        prev_year = str(year - 1) if str(year - 1) in pm_cols else None
        next_year = str(year + 1) if str(year + 1) in pm_cols else None
        
        # 计算相邻年份均值（至少有一个相邻年份有值才填充）
        fill_values = []
        if prev_year:
            prev_val = df[(df["city"] == city)][prev_year].iloc[0]
            if not pd.isnull(prev_val):
                fill_values.append(prev_val)
        if next_year:
            next_val = df[(df["city"] == city)][next_year].iloc[0]
            if not pd.isnull(next_val):
                fill_values.append(next_val)
        
        if fill_values:
            df.loc[idx, col] = np.mean(fill_values)  # 用均值填充
        else:
            df.loc[idx, col] = df[col].median()  # 无相邻年份→用全局中位数填充

# 删除“城市/国家缺失”的行（分类列缺失无法分析）
df = df.dropna(subset=["city", "country"])

print(f"\n=== 缺失值处理结果 ===")
print(f"处理后总缺失值：{df.isnull().sum().sum()} 个")
print(f"处理后数据形状：{df.shape}（行×列）")













# 1. 删除“PM2.5为负”的无效值（浓度不可能为负）
for col in pm_cols:
    df = df[df[col] >= 0]

# 2. 用IQR法剔除极端异常值（避免个别异常拉高统计结果）
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)  # 下四分位数
    q3 = df[col].quantile(0.75)  # 上四分位数
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr  # 下界
    upper = q3 + 1.5 * iqr  # 上界
    return df[(df[col] >= lower) & (df[col] <= upper)]

# 对2017-2023年PM2.5列批量剔除异常值
for col in pm_cols:
    df = remove_outliers(df, col)

print(f"\n=== 异常值处理结果 ===")
print(f"处理后数据形状：{df.shape}（行×列）")
print(f"处理后2023年PM2.5范围：{df['2023'].min():.2f} ~ {df['2023'].max():.2f} μg/m³")







# 重复定义：同一城市（city）的记录→保留第一条
df = df.drop_duplicates(subset=["city"], keep="first")

print(f"\n=== 重复值处理结果 ===")
print(f"最终数据形状：{df.shape}（行×列）")
print(f"最终覆盖城市数：{df['city'].nunique()} 个")






















# 1. 计算“全球PM2.5年均值”（所有城市均值）→ 看整体趋势
global_yearly_avg = df[pm_cols].mean().reset_index()  # 按年份计算全球均值
global_yearly_avg.columns = ["year", "pm25_avg"]  # 重命名列（方便后续使用）
global_yearly_avg["year"] = global_yearly_avg["year"].astype(int)  # 年份转为整数

print("=== 2017-2023年全球PM2.5年均值（μg/m³）===")
print(global_yearly_avg.round(2))

# 2. 计算“各国PM2.5年均值”→ 看不同国家的趋势
country_yearly_avg = df.groupby("country")[pm_cols].mean().reset_index()
# 转换为“长格式”（更适合分析：每行是“国家+年份+PM2.5”）
country_yearly_long = pd.melt(
    country_yearly_avg,
    id_vars=["country"],  # 分组列：国家
    value_vars=pm_cols,  # 数值列：2017-2023
    var_name="year",     # 新列名：年份
    value_name="pm25_avg"# 新列名：PM2.5均值
)
country_yearly_long["year"] = country_yearly_long["year"].astype(int)

# 查看前3个国家的年度趋势
print("\n=== 部分国家2017-2023年PM2.5年均值（μg/m³）===")
sample_countries = df["country"].unique()[:3]  # 取前3个国家
print(country_yearly_long[country_yearly_long["country"].isin(sample_countries)].round(2))

# 3. 计算“各国PM2.5改善率”→ 看2023年比2017年的变化（改善/恶化）
country_improve = df.groupby("country")[["2017", "2023"]].mean().reset_index()
country_improve["improve_rate"] = (country_improve["2023"] - country_improve["2017"]) / country_improve["2017"] * 100
country_improve.columns = ["country", "pm25_2017", "pm25_2023", "improve_rate"]  # 重命名
country_improve["improve_rate"] = country_improve["improve_rate"].round(2)  # 保留2位小数

print("\n=== 各国2017-2023年PM2.5改善率（%，负号表示改善）===")
print(country_improve.sort_values("improve_rate").round(2))  # 按改善率升序（改善越多越靠前）
























# 1. 国家层面：2023年PM2.5均值排名（Top10污染最严重 + Top10最清洁）
country_2023 = df.groupby("country")["2023"].mean().reset_index()
country_2023.columns = ["country", "pm25_2023"]
country_2023 = country_2023.sort_values("pm25_2023", ascending=False)  # 降序

print("\n=== 2023年PM2.5均值Top10污染国家（μg/m³）===")
print(country_2023.head(10).round(2))

print("\n=== 2023年PM2.5均值Top10清洁国家（μg/m³）===")
print(country_2023.tail(10).round(2))

# 2. 城市层面：2023年PM2.5均值排名（Top10污染 + Top10清洁，带国家信息）
city_2023 = df[["city", "country", "2023"]].sort_values("2023", ascending=False)
city_2023.columns = ["city", "country", "pm25_2023"]

print("\n=== 2023年PM2.5均值Top10污染城市（μg/m³）===")
print(city_2023.head(10).round(2))

print("\n=== 2023年PM2.5均值Top10清洁城市（μg/m³）===")
print(city_2023.tail(10).round(2))

# 3. 国家内部城市差异：计算每个国家“城市PM2.5标准差”（差异越大，城市间污染不均越严重）
country_city_var = df.groupby("country")["2023"].agg(["mean", "std"]).reset_index()
country_city_var.columns = ["country", "pm25_mean", "pm25_std"]
country_city_var["std_ratio"] = (country_city_var["pm25_std"] / country_city_var["pm25_mean"] * 100).round(2)  # 标准差占均值比例

print("\n=== 各国城市间PM2.5差异（2023年，标准差占均值比例%）===")
print(country_city_var.sort_values("std_ratio", ascending=False).round(2))
























plt.figure(figsize=(12, 6))
# 绘制全球均值折线（带标记点）
plt.plot(
    global_yearly_avg["year"], 
    global_yearly_avg["pm25_avg"], 
    marker="o", linewidth=2, color="#E74C3C", 
    label=f"全球均值（{global_yearly_avg['pm25_avg'].mean():.2f} μg/m³）"
)

# 叠加3个典型国家的趋势（可选，避免图表杂乱）
for country in sample_countries:
    country_data = country_yearly_long[country_yearly_long["country"] == country]
    plt.plot(
        country_data["year"], 
        country_data["pm25_avg"], 
        marker="s", linewidth=1.5, 
        label=f"{country}（{country_data['pm25_avg'].mean():.2f} μg/m³）"
    )

# 图表美化
plt.title("2017-2023年全球及典型国家PM2.5年均浓度变化趋势", fontsize=14, pad=20)
plt.xlabel("年份", fontsize=12)
plt.ylabel("PM2.5浓度（μg/m³）", fontsize=12)
plt.xticks(global_yearly_avg["year"])  # x轴刻度：2017-2023
plt.grid(alpha=0.3)  # 网格透明度
plt.legend(loc="best")  # 图例位置
plt.savefig("global_pm25_trend.png", dpi=300, bbox_inches="tight")  # 保存图片
plt.show()





















plt.figure(figsize=(14, 8))
# 取Top10污染城市
top10_cities = city_2023.head(10)
# 绘制水平柱状图（避免城市名重叠）
bars = plt.barh(
    y=range(len(top10_cities)), 
    width=top10_cities["pm25_2023"], 
    color="#3498DB"
)

# 在柱子上添加数值标签
for i, (idx, row) in enumerate(top10_cities.iterrows()):
    plt.text(
        row["pm25_2023"] + 1,  # 文字在柱子右侧
        i, 
        f"{row['city']}（{row['country']}）：{row['pm25_2023']:.2f}", 
        va="center", fontsize=10
    )

# 图表美化
plt.title("2023年PM2.5均值Top10污染城市", fontsize=14, pad=20)
plt.xlabel("PM2.5浓度（μg/m³）", fontsize=12)
plt.ylabel("城市（国家）", fontsize=12)
plt.yticks([])  # 隐藏y轴刻度（已在柱子上标注城市名）
plt.grid(alpha=0.3, axis="x")  # 仅x轴显示网格
plt.savefig("top10_polluted_cities.png", dpi=300, bbox_inches="tight")
plt.show()







# 筛选有完整2017-2023数据的国家（至少5个国家才画热力图）
if len(country_improve) >= 5:
    # 转换为“国家×年份”透视表（2017和2023年）
    improve_pivot = country_improve.pivot_table(
        index="country",
        values=["pm25_2017", "pm25_2023"]
    )
    
    plt.figure(figsize=(10, 8))
    # 绘制热力图
    sns.heatmap(
        improve_pivot,
        annot=True,        # 显示数值
        fmt=".2f",         # 保留2位小数
        cmap="YlOrRd_r",   # 颜色反转（数值越小越绿，代表越清洁）
        cbar_kws={"label": "PM2.5浓度（μg/m³）"}
    )
    plt.title("各国2017-2023年PM2.5浓度对比（绿色表示改善）", fontsize=14, pad=20)
    plt.savefig("country_pm25_improve.png", dpi=300, bbox_inches="tight")
    plt.show()






# 自动生成关键结论（可直接复制到报告）
print("\n=== 数据分析核心结论 ===")
# 1. 时间趋势：
print(f"1. 时间趋势：2017-2023年全球PM2.5年均浓度从 {global_yearly_avg[global_yearly_avg['year']==2017]['pm25_avg'].iloc[0]:.2f} μg/m³ 降至 {global_yearly_avg[global_yearly_avg['year']==2023]['pm25_avg'].iloc[0]:.2f} μg/m³，累计变化 {global_yearly_avg[global_yearly_avg['year']==2023]['pm25_avg'].iloc[0] - global_yearly_avg[global_yearly_avg['year']==2017]['pm25_avg'].iloc[0]:.2f} μg/m³（{'改善' if global_yearly_avg[global_yearly_avg['year']==2023]['pm25_avg'].iloc[0] < global_yearly_avg[global_yearly_avg['year']==2017]['pm25_avg'].iloc[0] else '恶化'}）。")

# 2. 空间差异（国家）
most_polluted_country = country_2023.iloc[0]
cleanest_country = country_2023.iloc[-1]
print(f"2. 国家差异：2023年PM2.5最严重的国家是 {most_polluted_country['country']}（{most_polluted_country['pm25_2023']:.2f} μg/m³），最清洁的是 {cleanest_country['country']}（{cleanest_country['pm25_2023']:.2f} μg/m³），差距达 {most_polluted_country['pm25_2023'] - cleanest_country['pm25_2023']:.2f} μg/m³。")

# 3. 空间差异（城市）
most_polluted_city = city_2023.iloc[0]
cleanest_city = city_2023.iloc[-1]
print(f"3. 城市差异：2023年PM2.5最严重的城市是 {most_polluted_city['city']}（{most_polluted_city['country']}，{most_polluted_city['pm25_2023']:.2f} μg/m³），最清洁的是 {cleanest_city['city']}（{cleanest_city['country']}，{cleanest_city['pm25_2023']:.2f} μg/m³）。")

# 4. 改善效果
best_improve = country_improve.sort_values("improve_rate").iloc[0]
worst_improve = country_improve.sort_values("improve_rate").iloc[-1]
print(f"4. 改善效果：2017-2023年PM2.5改善最显著的国家是 {best_improve['country']}（改善 {abs(best_improve['improve_rate']):.2f}%），恶化最严重的是 {worst_improve['country']}（恶化 {worst_improve['improve_rate']:.2f}%）。")











