"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""
import pandas as pd
import config, os
from sklearn.externals.joblib import dump
from sklearn.preprocessing import StandardScaler

class FeatureDictionary(object):
    def __init__(self,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[], scale = False):
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.scale = scale
        self.gen_feat_dict()

    def gen_feat_dict(self):
        df = pd.concat([self.dfTrain, self.dfTest])
        if not self.scale:
            print ('start to scaler data..')
            df_numerical = df[config.NUMERIC_COLS].astype('float32')
            self.scale = StandardScaler()
            self.scale.fit(df_numerical)
            dump(self.scale, os.path.join(config.sub_dir, 'scale.bin'))
            print ('scaler finished.')
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict, scale):
        self.feat_dict = feat_dict
        self.scale = scale

    def parse(self, df=None, has_label=False):
        dfi = df.copy()
        if has_label:
            y = dfi[config.LABEL].values.tolist()
            dfi.drop(config.IGNORE_COLS, axis=1, inplace=True)
        else:
            ids = dfi["pin"].values.tolist()
            dfi.drop(["pin"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in config.IGNORE_COLS:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in config.NUMERIC_COLS:
                dfi[col] = self.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict[col])
                dfi[col] = dfi[col].fillna(0)
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values
        numeric_col = dfv[config.NUMERIC_COLS].astype('float32')
        numeric_col_list = self.scale.transform(numeric_col)
        NUMERIC_COLS_ID = dict((k,v) for v,k in enumerate(config.NUMERIC_COLS))
        all_colums = dfv.columns.tolist()
        for ind in range(len(all_colums)):
            if all_colums[ind] in config.NUMERIC_COLS:
                Xv[:,ind] = numeric_col_list[:,NUMERIC_COLS_ID[all_colums[ind]]]
        Xv = Xv.tolist()

        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids