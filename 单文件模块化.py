
# 对原本的代码进行了模块化管理，方便维护和拓展。每个功能块都封装成独立函数，主函数串联整个流程

# 1. 导入依赖（统一放顶部）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm



#  全局配置函数
def setup_global_config():
    # 直接指定微软软黑文字的路径
    font_path = "c:\Windows\Fonts\STXIHEI.TTF"  
    os.path.exists(font_path)
        # 加载字体文件
    font_prop = fm.FontProperties(fname=font_path)
        # 全局设置：所有文本都用这个字体
    plt.rcParams['font.family'] = font_prop.get_name()




#  数据加载函数
def load_data(file_path="air_pollution.csv"):
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件{file_path}不存在，请检查路径！")
    except UnicodeDecodeError:
        # 兼容常见编码
        df = pd.read_csv(file_path, encoding="gbk")
    
    # 校验必要列
    required_cols = ["city", "country"] + [str(y) for y in range(2017,2024)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据缺失必要列：{missing_cols}")
    return df




#  缺失值处理函数
def clean_missing_values(df, pm_cols):
    #  删除两列存在缺失值的整行数据
    df = df.dropna(subset=["city", "country"])
    
    #  分层填充（优先级：城市均值 > 国家均值 > 全局中位数）
    for col in pm_cols:
        # 第一步：用该城市的均值填充（最贴合）
        city_mean = df.groupby("city")[col].transform("mean")
        df[col] = df[col].fillna(city_mean)
        
        # 第二步：若城市均值仍缺失，用该国家的均值填充
        country_mean = df.groupby("country")[col].transform("mean")
        df[col] = df[col].fillna(country_mean)
        
        # 第三步：若国家均值仍缺失，用全局中位数填充
        df[col] = df[col].fillna(df[col].median())
    
    # 验证缺失值
    print(f"缺失值处理后：总缺失值 {df.isnull().sum().sum()} 个")
    return df




#  统计分析函数
def analyze_data(df, pm_cols):
    #  全球年度均值
    global_yearly_avg = df[pm_cols].mean().reset_index()
    global_yearly_avg.columns = ["year", "pm25_avg"]
    global_yearly_avg["year"] = global_yearly_avg["year"].astype(int)
    print("=== 2017-2023年全球PM2.5年均值（μg/m³）===")
    print(global_yearly_avg.round(2))
    
    #  国家年度均值
    country_yearly_avg = df.groupby("country")[pm_cols].mean().reset_index()
    country_yearly_long = pd.melt(
        country_yearly_avg, id_vars=["country"], value_vars=pm_cols,
        var_name="year", value_name="pm25_avg"
    )
    country_yearly_long["year"] = country_yearly_long["year"].astype(int)
    sample_countries = df["country"].unique()[:3]
    print("\n=== 部分国家2017-2023年PM2.5年均值（μg/m³）===")
    print(country_yearly_long[country_yearly_long["country"].isin(sample_countries)].round(2))
    
    #  国家改善率
    country_improve = df.groupby("country")[["2017", "2023"]].mean().reset_index()
    country_improve["improve_rate"] = (country_improve["2023"] - country_improve["2017"]) / country_improve["2017"] * 100
    country_improve.columns = ["country", "pm25_2017", "pm25_2023", "improve_rate"]
    country_improve["improve_rate"] = country_improve["improve_rate"].round(2)
    print("\n=== 各国2017-2023年PM2.5改善率（%，负号表示改善）===")
    print(country_improve.sort_values("improve_rate").round(2))
    
    #  国家/城市排名
    country_2023 = df.groupby("country")["2023"].mean().reset_index().sort_values("2023", ascending=False)
    country_2023.columns = ["country", "pm25_2023"]
    print("\n=== 2023年PM2.5均值Top10污染国家（μg/m³）===")
    print(country_2023.head(10).round(2))
    print("\n=== 2023年PM2.5均值Top10清洁国家（μg/m³）===")
    print(country_2023.tail(10).round(2))
    
    city_2023 = df[["city", "country", "2023"]].sort_values("2023", ascending=False)
    city_2023.columns = ["city", "country", "pm25_2023"]
    print("\n=== 2023年PM2.5均值Top10污染城市（μg/m³）===")
    print(city_2023.head(10).round(2))
    print("\n=== 2023年PM2.5均值Top10清洁城市（μg/m³）===")
    print(city_2023.tail(10).round(2))
    
    return global_yearly_avg, country_yearly_long, country_improve, country_2023, city_2023





#  可视化函数
def visualize_data(global_yearly_avg, country_yearly_long, country_improve, city_2023):
    """生成所有可视化图表"""
    sample_countries = country_yearly_long["country"].unique()[:3]
    
    #  全球+典型国家趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(
        global_yearly_avg["year"], global_yearly_avg["pm25_avg"],
        marker="o", linewidth=2, color="#E74C3C",
        label=f"全球均值（{global_yearly_avg['pm25_avg'].mean():.2f} μg/m³）"
    )
    for country in sample_countries:
        country_data = country_yearly_long[country_yearly_long["country"] == country]
        plt.plot(
            country_data["year"], country_data["pm25_avg"],
            marker="s", linewidth=1.5,
            label=f"{country}（{country_data['pm25_avg'].mean():.2f} μg/m³）"
        )
    plt.title("2017-2023年全球及典型国家PM2.5年均浓度变化趋势", fontsize=14, pad=20)
    plt.xlabel("年份", fontsize=12)
    plt.ylabel("PM2.5浓度（μg/m³）", fontsize=12)
    plt.xticks(global_yearly_avg["year"])
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.savefig("global_pm25_trend.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    #  Top10污染城市柱状图
    plt.figure(figsize=(14, 8))
    top10_cities = city_2023.head(10)
    bars = plt.barh(range(len(top10_cities)), width=top10_cities["pm25_2023"], color="#3498DB")
    for i, (idx, row) in enumerate(top10_cities.iterrows()):
        plt.text(
            row["pm25_2023"] + 1, i,
            f"{row['city']}（{row['country']}）：{row['pm25_2023']:.2f}",
            va="center", fontsize=10
        )
    plt.title("2023年PM2.5均值Top10污染城市", fontsize=14, pad=20)
    plt.xlabel("PM2.5浓度（μg/m³）", fontsize=12)
    plt.ylabel("城市（国家）", fontsize=12)
    plt.yticks([])
    plt.grid(alpha=0.3, axis="x")
    plt.savefig("top10_polluted_cities.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    # 6.3 国家改善热力图
    if len(country_improve) >= 5:
        improve_pivot = country_improve.pivot_table(
            index="country", values=["pm25_2017", "pm25_2023"]
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            improve_pivot, annot=True, fmt=".2f",
            cmap="YlOrRd_r", cbar_kws={"label": "PM2.5浓度（μg/m³）"}
        )
        plt.title("各国2017-2023年PM2.5浓度对比（绿色表示改善）", fontsize=14, pad=20)
        plt.savefig("country_pm25_improve.png", dpi=300, bbox_inches="tight")
        plt.show()





#  结论生成函数
def generate_conclusion(global_yearly_avg, country_2023, city_2023, country_improve):
    """自动生成核心结论"""
    print("\n=== 数据分析核心结论 ===")
    # 时间趋势
    avg_2017 = global_yearly_avg[global_yearly_avg["year"]==2017]["pm25_avg"].iloc[0]
    avg_2023 = global_yearly_avg[global_yearly_avg["year"]==2023]["pm25_avg"].iloc[0]
    trend = "改善" if avg_2023 < avg_2017 else "恶化"
    print(f"1. 时间趋势：2017-2023年全球PM2.5年均浓度从 {avg_2017:.2f} μg/m³ 降至 {avg_2023:.2f} μg/m³，累计变化 {avg_2023 - avg_2017:.2f} μg/m³（{trend}）。")
    
    # 国家差异
    most_polluted_country = country_2023.iloc[0]
    cleanest_country = country_2023.iloc[-1]
    print(f"2. 国家差异：2023年PM2.5最严重的国家是 {most_polluted_country['country']}（{most_polluted_country['pm25_2023']:.2f} μg/m³），最清洁的是 {cleanest_country['country']}（{cleanest_country['pm25_2023']:.2f} μg/m³），差距达 {most_polluted_country['pm25_2023'] - cleanest_country['pm25_2023']:.2f} μg/m³。")
    
    # 城市差异
    most_polluted_city = city_2023.iloc[0]
    cleanest_city = city_2023.iloc[-1]
    print(f"3. 城市差异：2023年PM2.5最严重的城市是 {most_polluted_city['city']}（{most_polluted_city['country']}，{most_polluted_city['pm25_2023']:.2f} μg/m³），最清洁的是 {cleanest_city['city']}（{cleanest_city['country']}，{cleanest_city['pm25_2023']:.2f} μg/m³）。")
    
    # 改善效果
    best_improve = country_improve.sort_values("improve_rate").iloc[0]
    worst_improve = country_improve.sort_values("improve_rate").iloc[-1]
    print(f"4. 改善效果：2017-2023年PM2.5改善最显著的国家是 {best_improve['country']}（改善 {abs(best_improve['improve_rate']):.2f}%），恶化最严重的是 {worst_improve['country']}（恶化 {worst_improve['improve_rate']:.2f}%）。")







#  主函数（入口）
def main():
    # 步骤1：全局配置
    setup_global_config()
    # 步骤2：加载数据
    df = load_data()

    # 步骤3：定义年份列（pm_cols），再执行数据清洗
    pm_cols = [str(year) for year in range(2017, 2024)]  # 新增：定义2017-2023年份列
    df_clean = clean_missing_values(df, pm_cols)  

    # 步骤4：统计分析
    global_avg, country_yearly, country_improve, country_2023, city_2023 = analyze_data(df_clean, pm_cols)
    
    # 步骤5：可视化
    visualize_data(global_avg, country_yearly, country_improve, city_2023)
    # 步骤6：生成结论
    generate_conclusion(global_avg, country_2023, city_2023, country_improve)





# 执行主函数
if __name__ == "__main__":
    main()





















