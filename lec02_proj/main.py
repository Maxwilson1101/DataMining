import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


def inspect_data(df):
    df = pd.array([1, 2 , 3])


def clean_data(df):
    pass


def show_customer_stats(df):
    pass


def show_total_cost_stats(df):
    pass


def show_trend_by_country(df):
    pass


# 可视化结果
# 堆叠柱状图
month_country_count_df.plot(kind='bar', stacked=True, rot=45)
plt.xlabel('Month')
plt.ylabel('#Transaction')
plt.tight_layout()
plt.savefig('./output/country_trend_stacked_bar.png')
plt.show()

# 热图
sns.heatmap(month_country_count_df.T)
plt.xlabel('Month')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('./output/country_trend_heatmap.png')
plt.show()


def main():
    """
        主函数
    """
    if not os.path.exists(CLN_DATA_FILE):
        # 如果不存在清洗后的数据集，进行数据清洗
        raw_data_df = pd.read_excel(RAW_DATA_FILE, dtype={'InvoiceNo': str,
                                                          'StockCode': str,
                                                          'CustomerID': str})

        # 查看数据集信息
        inspect_data(raw_data_df)

        # 数据清洗
        cln_data_df = clean_data(raw_data_df)
    else:
        print('读取已清洗的数据')

        cln_data_df = pd.read_csv(CLN_DATA_FILE)

    # 数据分析
    # 1. 比较各国家的客户数
    show_customer_stats(cln_data_df)

    # 2. 比较各国家的成交额
    show_total_cost_stats(cln_data_df)

    # 3. 统计各国家交易记录的趋势
    show_trend_by_country(cln_data_df)


if __name__ == '__main__':
    main()
