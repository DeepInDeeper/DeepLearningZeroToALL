# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


def ensemble():
    stacked_1 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.166586_submission.csv')
    stacked_2 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.167542_submission.csv')
    stacked_3 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.167621_submission.csv')
    stacked_4 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.166586_submission.csv')
    stacked_5 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.168133_submission.csv')
    stacked_6 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.170522_submission.csv')
    stacked_7 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.180278_submission.csv')
    stacked_8 = pd.read_csv('./pth/kfold/' + 'ResNetLike_0.185876_submission.csv')
    sub = pd.DataFrame()
    sub['id'] = stacked_1['id']
    sub['is_iceberg'] = np.exp(np.mean(
        [
            stacked_1['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_2['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_3['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_4['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_5['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_6['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_7['is_iceberg'].apply(lambda x: np.log(x)), \
            stacked_8['is_iceberg'].apply(lambda x: np.log(x)), \
            ], axis=0))
    sub.to_csv('./pth/kfold/' + 'ensamble.csv', index=False, float_format='%.6f')