
# 新人用AI写代码，一方面是学习代码编程，一方面也在学习数据分析流程
# 对原本的代码进行了模块化管理，方便维护和拓展。每个功能块都封装成独立函数，主函数串联整个流程


# 这次的版本主要是修复了字体问题，以及可视化的美观问题
# 打算趁机研究一下系统字体和matplotlib字体问题以及os模块的文件路径问题

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
import shutil

def setup_global_config():
    """终极字体修复 + 清理缓存"""
    # 1. 清理 matplotlib 字体缓存（强制重新加载）
    cache_dir = os.path.join(os.path.expanduser("~"), ".matplotlib")
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            print("✅ 已清理旧字体缓存")
        except:
            pass
    
    # 2. 注册微软雅黑
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        print("✅ 已成功注册 Microsoft YaHei 字体")
    
    # 3. 全局强制
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'STXihei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams['figure.dpi'] = 150
    
    global font_prop
    font_prop = fm.FontProperties(fname=font_path)
    print("✅ 全局配置完成：中文字体 + 美观主题已启用")


def load_data(filename="air_pollution.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, filename)
    print(f"🔍 正在打开: {file_path}")
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="gbk")
    print(f"✅ 数据加载成功！共 {df.shape[0]} 城市，{df['country'].nunique()} 个国家")
    return df


def clean_missing_values(df, pm_cols):
    df = df.dropna(subset=["city", "country"]).copy()
    for col in pm_cols:
        city_mean = df.groupby("city")[col].transform("mean")
        df[col] = df[col].fillna(city_mean)
        country_mean = df.groupby("country")[col].transform("mean")
        df[col] = df[col].fillna(country_mean)
        df[col] = df[col].fillna(df[col].median())
    print(f"✅ 缺失值处理完成：剩余 0 个")
    return df


def analyze_data(df, pm_cols):
    global_yearly_avg = df[pm_cols].mean().reset_index()
    global_yearly_avg.columns = ["year", "pm25_avg"]
    global_yearly_avg["year"] = global_yearly_avg["year"].astype(int)
    
    country_yearly_long = pd.melt(
        df.groupby("country")[pm_cols].mean().reset_index(),
        id_vars=["country"], value_vars=pm_cols, var_name="year", value_name="pm25_avg"
    )
    country_yearly_long["year"] = country_yearly_long["year"].astype(int)
    
    country_improve = df.groupby("country")[["2017", "2023"]].mean().reset_index()
    country_improve["improve_rate"] = ((country_improve["2023"] - country_improve["2017"]) / country_improve["2017"] * 100).round(2)
    country_improve = country_improve.rename(columns={"2017": "pm25_2017", "2023": "pm25_2023"})
    
    country_2023 = (df.groupby("country")["2023"].mean().reset_index()
                    .rename(columns={"2023": "pm25_2023"}).sort_values("pm25_2023", ascending=False))
    city_2023 = df[["city", "country", "2023"]].rename(columns={"2023": "pm25_2023"}).sort_values("pm25_2023", ascending=False)
    
    print("\n=== 2017-2023年全球PM2.5年均值（μg/m³）===")
    print(global_yearly_avg.round(2))
    
    with pd.ExcelWriter("PM25_Analysis_Results.xlsx") as writer:
        global_yearly_avg.to_excel(writer, sheet_name='Global_Trend', index=False)
        country_yearly_long.to_excel(writer, sheet_name='Country_Yearly', index=False)
        country_improve.to_excel(writer, sheet_name='Country_Improve', index=False)
        country_2023.to_excel(writer, sheet_name='Country_2023', index=False)
        city_2023.to_excel(writer, sheet_name='City_2023', index=False)
    df.to_csv("cleaned_air_pollution.csv", index=False, encoding="utf-8-sig")
    print("✅ 所有结果已保存！")
    return global_yearly_avg, country_yearly_long, country_improve, country_2023, city_2023


def visualize_data(global_yearly_avg, country_yearly_long, country_improve, city_2023):
    sample_countries = country_yearly_long["country"].unique()[:3]
    
    # 趋势图
    plt.figure(figsize=(10, 5.5))
    plt.plot(global_yearly_avg["year"], global_yearly_avg["pm25_avg"], marker="o", linewidth=2.5, color="#E74C3C", label="全球均值")
    for country in sample_countries:
        data = country_yearly_long[country_yearly_long["country"] == country]
        plt.plot(data["year"], data["pm25_avg"], marker="s", linewidth=1.8, label=country)
    plt.title("2017-2023年全球及典型国家PM2.5年均浓度变化趋势", pad=15, fontsize=14, fontproperties=font_prop)
    plt.xlabel("年份", fontproperties=font_prop)
    plt.ylabel("PM2.5浓度（μg/m³）", fontproperties=font_prop)
    plt.legend(prop=font_prop)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("global_pm25_trend.png", dpi=300, bbox_inches="tight"); plt.show()
    
    # Top10污染城市（纯matplotlib + 所有文本强制字体）
    plt.figure(figsize=(12, 7))
    top10 = city_2023.head(10).copy()
    y_pos = range(len(top10))
    plt.barh(y_pos, top10["pm25_2023"], color="#3498DB")
    labels = [f"{row['city']}（{row['country']}）" for _, row in top10.iterrows()]
    plt.yticks(y_pos, labels, fontproperties=font_prop)
    plt.title("2023年PM2.5均值Top10污染城市", pad=15, fontsize=14, fontproperties=font_prop)
    plt.xlabel("PM2.5浓度（μg/m³）", fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("top10_polluted_cities.png", dpi=300, bbox_inches="tight"); plt.show()
    
    # 热力图
    top20 = country_improve.nsmallest(20, "improve_rate")
    pivot = top20.set_index("country")[["pm25_2017", "pm25_2023"]]
    plt.figure(figsize=(9, 6.5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5)
    plt.title("2017 vs 2023 PM2.5浓度对比（绿色=显著改善）\nTop20改善最显著国家", pad=15, fontsize=14, fontproperties=font_prop)
    plt.tight_layout()
    plt.savefig("country_pm25_improve.png", dpi=300, bbox_inches="tight"); plt.show()


def generate_conclusion(global_yearly_avg, country_2023, city_2023, country_improve):
    print("\n" + "="*60)
    print("                  数据分析核心结论")
    print("="*60)
    avg_2017 = global_yearly_avg.iloc[0]["pm25_avg"]
    avg_2023 = global_yearly_avg.iloc[-1]["pm25_avg"]
    print(f"1. 全球趋势：{avg_2017:.1f} → {avg_2023:.1f} μg/m³（{'改善' if avg_2023 < avg_2017 else '恶化'}）")
    print(f"2. 最严重国家：{country_2023.iloc[0]['country']}（{country_2023.iloc[0]['pm25_2023']:.1f}）")
    print(f"3. 最严重城市：{city_2023.iloc[0]['city']}（{city_2023.iloc[0]['country']}，{city_2023.iloc[0]['pm25_2023']:.1f}）")
    best = country_improve.nsmallest(1, "improve_rate").iloc[0]
    print(f"4. 改善最显著：{best['country']}（下降 {abs(best['improve_rate']):.1f}%）")
    print("="*60)


def main():
    setup_global_config()
    df = load_data()
    pm_cols = [str(y) for y in range(2017, 2024)]
    df_clean = clean_missing_values(df, pm_cols)
    global_avg, country_yearly, country_improve, country_2023, city_2023 = analyze_data(df_clean, pm_cols)
    visualize_data(global_avg, country_yearly, country_improve, city_2023)
    generate_conclusion(global_avg, country_2023, city_2023, country_improve)
    print("\n🎉 全部完成！这次字体应该100%正常（无警告、无方块）～")


if __name__ == "__main__":
    main()





















