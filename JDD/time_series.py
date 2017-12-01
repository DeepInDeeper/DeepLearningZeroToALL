#coding:utf-8
import pandas as pd
import numpy as np

order = pd.read_csv(r'/media/yijie/文档/dataset/Sales_Forecast_Qualification/t_order.csv')
order = order[order.sale_amt >= 0]
order['datetime'] = pd.to_datetime(order['ord_dt'])
sale = order.groupby('shop_id').resample('1M', key='datetime').sum()
del sale['shop_id']
sale = sale.reset_index()
sale['month'] = sale['datetime'].map(lambda x: x.month)
sale['ratio'] = sale['month']
# 参数来自全部店铺每个月的销量除以平均销量
sale['ratio'].replace(8, 0.754436, inplace=True)
sale['ratio'].replace(9, 0.991596, inplace=True)
sale['ratio'].replace(10, 1.247299, inplace=True)
sale['ratio'].replace(11, 1.495532, inplace=True)
sale['ratio'].replace(12, 1.141776, inplace=True)
sale['ratio'].replace(1, 0.697440, inplace=True)
sale['ratio'].replace(2, 0.691003, inplace=True)
sale['ratio'].replace(3, 1.023496, inplace=True)
sale['ratio'].replace(4, 0.957429, inplace=True)
submission = []
for shop in range(1, 3001):
    print('round %d/3000' % shop)
    sale_i_df = sale[sale['shop_id'] == shop][['sale_amt', 'ratio']]
    len_of_sale_i = sale_i_df.shape[0]
    loss_May, loss_June, loss_July = [], [], []
    list_of_values = list(range(int(sale_i_df['sale_amt'].min()), int(sale_i_df['sale_amt'].max()) + 1, 50))
    sale_i_df['sale_amt'] /= sale_i_df['ratio']
    pre_May, pre_June, pre_July = list_of_values, list_of_values, list_of_values
    for pre_sale in list_of_values:
        sale_i_df['sale_amt_diff'] = np.abs(pre_sale - sale_i_df['sale_amt'])
        sale_i_df['decay'] = list(range(len_of_sale_i, 0, -1))
        loss_May.append((sale_i_df['sale_amt_diff'] / sale_i_df['decay']).sum())
        loss_June.append((sale_i_df['sale_amt_diff'] / (sale_i_df['decay'] + 1)).sum())
        loss_July.append((sale_i_df['sale_amt_diff'] / (sale_i_df['decay'] + 2)).sum())
    submission.append([shop, pre_May[np.argmin(loss_May)] + 1.1 * pre_June[np.argmin(loss_June)] + pre_July[np.argmin(loss_July)]])
sub = pd.DataFrame(submission)
sub.to_csv(r'./submission.csv', index=None, header=None)
