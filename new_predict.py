from sklearn.preprocessing import OneHotEncoder
import joblib
import xgboost as xgb

import pandas as pd
import numpy as np
import json
import os
import pdb
import copy

current_path = os.path.dirname(os.path.abspath(__file__))   # refers to application_top

class modelSalesFridge(object):
    def __init__(self,model_dir = None):
        self.product_name = "fridge"
        self.model_dir = model_dir if model_dir else os.path.dirname(os.path.abspath(__file__)) + "/model_dir"
        self.brand_cnt = 50
        self.color_cnt = 5
        # 清洗后的数据，可以不作任何处理直接使用的特征
        self.numerical_x = [
            "sale_mode",
            "height", "width", "depth", "volume", "display_type", "screen_diagonal",

            "efficacy", "efficiency_volumen_rate", "eval_by_noise",
            "vegetable_preservation_cap", "energy_efficiency", "cycle_cnt",

            "wifi", "ice_making", "dry_wet_storage", "wide_temp_change", "sterilization",
            "low_temp_compensation", "environmentally_friendly_liner",
            "automatic_hovering_door", "90degree_opening", "fully_open_drawer",
            "zero_preservation", "micro_frozen_preserve", "cell_grade_fresh",
            'defrost','compressor','tem_control'
        ]
        # UV与属性之间的单调性约束
        self.constraint = {
            "volume": 1, "display_type": 1,"efficacy":-1,'eval_by_noise':-1,'cycle_cnt':1,
            "wifi": 1, "ice_making": 1, "dry_wet_storage": 1, "wide_temp_change": 1,
            "sterilization": 1, "low_temp_compensation": 1,
            "zero_preservation": 1, "micro_frozen_preserve": 1, "cell_grade_fresh": 1,
            "rule_based_price_avg":-1
        }
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)
        #生成存储模型的名字


    # ====================================================================================================
    # 对长尾特征进行截断处理
    # ====================================================================================================
    @staticmethod
    def truncate_feature_tail(original_feature, feature_cnt):
        feature_name = original_feature.name
        feature_info = original_feature.value_counts()
        if "其他" in feature_info:
            feature_info["其他"] = 0
        if "其它" in feature_info:
            feature_info["其它"] = 0
        feature_info = feature_info.reset_index().sort_values(by=[feature_name,'index'],ascending=False)
        feature_info.columns = [feature_name, "sku_cnt"]
        feature_info[feature_name + "_label"] = range(1, len(feature_info) + 1)
        feature_info[feature_name + "_label"] = feature_info[feature_name + "_label"].apply(
            lambda x: x if x <= feature_cnt else 0)
        feature_list = feature_info[feature_name][:feature_cnt].tolist()
        if feature_cnt < len(feature_info):
            feature_list.insert(0, "ohters")
        return feature_info[[feature_name, feature_name + "_label"]], feature_list

    # ===================================================================================================
    # 获得训练的特征，将从数据库读取的数据转换成训练模型的特征，并将特征处理过程中的数据存储下来
    # ===================================================================================================
    def get_train_feature(self,data):
        # (1) 定义onehot需要长尾截断的数值，当列值多余该值，则被截断，当列值少于该值则全部保留
        tail_columns = {
            "material": 100, "door_type": 40,  "freeze_type": 5,
             "color": self.color_cnt,"freeze_cap_100l":6,'brand_std':self.brand_cnt
        }
        column_name = {}  # 存储的为需要onehot的列名与转化之后的列名，字典key为字典编码之后的列名
        feature_dict = {}  # 保存onehot的列值

        # (2) 处理列值很多，需要根据长尾截断的列

        for key, value in tail_columns.items():
            column_info, feature_dict[key] = self.truncate_feature_tail(data[key], value)
            column_name[key + "_label"] = ["%s_%s" % (key, x) for x in feature_dict[key]]
            data = pd.merge(data, column_info, how='left', on=[key])

        selected_columns = []  # 最终生成的特征字段名，即训练是所用的 X

        # (3) 将0-n的值进行onehot
        for key, value in column_name.items():
            if len(value) < 3:
                selected_columns.append(key)
            else:
                
                onehot_value = OneHotEncoder().fit_transform(data[key].values.reshape(-1, 1)).toarray()
                onehot_df = pd.DataFrame(onehot_value, columns=value)
                data = pd.concat([data, onehot_df], axis=1)
                selected_columns += value

        # (4) 保存特征处理字典
        with open('sale.json', "w+", encoding="utf-8") as f:
            json.dump(feature_dict, f, ensure_ascii=False)
        # data["sale_mode"] = data["sale_mode"].apply(lambda x: 1 if x == 1 else 0)
        return data, self.numerical_x+selected_columns

    # ===================================================================================================
    # 获得训练的特征
    # ===================================================================================================
    def get_predict_feature(self,data):
        with open('sale.json', "r", encoding="utf-8") as f:
            feature_dict = json.load(f)     # 读取需要转换字段的字典

        selected_columns = []

        for key, value in feature_dict.items():
            data[key + "_label"] = data[key].map(dict(zip(value, range(len(value))))).fillna(0)
            if len(value) < 3:              # 针对类型列少于3个列，直接用转换后的数值
                selected_columns.append(key + "_label")
            else:
                for i in range(len(value)):
                    data['%s_%s' % (key, value[i])] = data[key + "_label"].apply(lambda x: 1 if x == i else 0)
                    selected_columns.append('%s_%s' % (key, value[i]))
        #pdb.set_trace()
        if 'sale_mode' in data.columns:
            data["sale_mode"] = data["sale_mode"].apply(lambda x: 1 if x == 1 else 0)
        else:
            data["sale_mode"] = 1
        return data, self.numerical_x+selected_columns
    
    
   


    # ===================================================================================================
    # 训练模型
    # ===================================================================================================
    def fit(self,input_data):
        
        
        original_data = copy.deepcopy(input_data)
        sale = 'sales'
        original_data.dropna(how = 'any',inplace = True)
        #print(original_data.columns)
        # original_data[self.numerical_x] = original_data[self.numerical_x].astype(float)
        train_feature, x_name = self.get_train_feature(original_data)
    
        #pdb.set_trace()
        # 加入价格列
        x_name.insert(0,'rule_based_price_avg')
        y_name = sale
        print(x_name)
        train_feature['defrost'] = train_feature['defrost'].apply(lambda x :1 if x=='智能除霜' else 0)
        train_feature['compressor'] = train_feature['compressor'].apply(lambda x :1 if x=='变频' else 0)
        train_feature['tem_control'] = train_feature['tem_control'].apply(lambda x :1 if x=='电脑控温' else 0)
        
        train_feature = train_feature.dropna(subset=[y_name])
        train_feature = train_feature[train_feature['sales']>=0]# 去掉uv人数太少导致uvrate不太有意义的样本
        # train_feature = train_feature[train_feature['sale']<=0.3]# 去掉uv失衡样本
        #pdb.set_trace()
        # train_feature = train_feature[train_feature['rule_based_price_avg']>=300]
        
        train_flag = np.array([True] * int(len(train_feature)))
        train_flag[:int(len(train_feature) * 0.25)] = False
        np.random.seed(2021)
        # np.random.shuffle(train_flag)
        test_flag = np.array(1 - train_flag, dtype=bool)
        # & (train_feature['sale_mode']==1)
        # train_flag = np.array(1 - test_flag, dtype=bool) 
        # train_flag = np.array([True] * int(len(train_feature))) #用所有样本做训练
        ratio = 0.85
        param = {
            'booster': 'gbtree', 'verbosity': 1, 'seed': 0,

            'learning_rate': 0.1,
            'max_depth': 5, 'min_split_loss': 0, 'min_child_weight': 1,
            'max_bin': 128,
            'subsample': ratio, 'colsample_bytree': ratio, 'colsample_bylevel': ratio, 'colsample_bynode': ratio,
            'reg_lambda': 1.0, 'reg_alpha': 0.2,
            'scale_pos_weight': 1.0,

            'objective': 'reg:linear', 'eval_metric': 'mae',
            # 'objective': 'reg:squarederror', 'eval_metric': 'mae',
            'n_estimators': 200,
            'monotone_constraints':str(tuple([
                self.constraint[item] if item in self.constraint else 0 for item in x_name
            ]))
        }
        model = xgb.XGBRegressor(**param)
        #pdb.set_trace()
        print(len(train_feature.loc[train_flag, x_name]))

        print(len(train_feature.loc[test_flag, x_name]))

        model.fit(train_feature.loc[train_flag, x_name], train_feature.loc[train_flag, y_name])

        y_train = train_feature.loc[train_flag, ['item_sku_id', y_name]]
        y_train['pred'] = model.predict(train_feature.loc[train_flag, x_name])
        train_precision = self.cal_precision(y_train,y_name,'pred')
        print('训练精确度为{0}'.format(train_precision))
        #pdb.set_trace()
        y_train.loc[y_train['pred']<0,'pred']=0 
        y_train['diff'] = (y_train['pred'] - y_train[y_name]).abs()
        y_train['wmape'] = y_train['diff'] / y_train[y_name]
        train_wmape = y_train['diff'].sum() / y_train[y_name].sum()
        train_error = (y_train['diff'] / (y_train[y_name]+1e-9)).mean()
        #print(y_train)
        print("For the sale, the train WMAPE is {0}, train MAPE is {1}".format(train_wmape, train_error))

        y_test = train_feature.loc[test_flag, ['item_sku_id', y_name]]
        y_test['pred'] = model.predict(train_feature.loc[test_flag, x_name])
        y_test.loc[y_test['pred']<0,'pred']=0 
        y_test['diff'] = (y_test['pred'] - y_test[y_name]).abs()
        y_test['wmape'] = y_test['diff'] / y_test[y_name]
        test_wmape = y_test['diff'].sum() / y_test[y_name].sum()
        test_error = (y_test['diff'] / (y_test[y_name]+1e-9)).mean()
        test_precision = self.cal_precision(y_test,y_name,'pred')
        print('训练精确度为{0}'.format(test_precision))
        print("For the sale, the test WMAPE is {0},test MAPE is {1}".format(test_wmape, test_error))

        joblib.dump(model,'new_sales_fridge.model')

        # return [self.feature_name[1:], '{0}_clf_uv_xgb.model'.format(self.model_save_name)]
        return
    # ===================================================================================================
    # 预测销量
    # ===================================================================================================
    def predict(self,data):
       

        predict_data, x_name = self.get_predict_feature(copy.deepcopy(data))
        predict_data['defrost'] = predict_data['defrost'].apply(lambda x :1 if x=='智能除霜' else 0)
        predict_data['compressor'] = predict_data['compressor'].apply(lambda x :1 if x== '变频' else 0)
        predict_data['tem_control'] = predict_data['tem_control'].apply(lambda x :1 if x== '电脑控温' else 0)
        price = 'rule_based_price_avg'
        x_name.insert(0,'rule_based_price_avg')

        model = joblib.load('new_sales_fridge.model')
        result = model.predict(predict_data[x_name])

        return result

    def cal_precision(self,data,lable,pred,rate=0.5):
        data_tmp =copy.deepcopy(data)
        data_tmp['cls_res'] = data_tmp.apply(lambda x : 1 if abs(x[lable] - x[pred])<rate*x[pred] else 0,axis = 1)
        res = len(data_tmp[data_tmp['cls_res']==1])/len(data_tmp)
        return res
    

if __name__ == "__main__":
    new = modelSalesFridge()
    newsale = pd.read_csv('newsale.csv')
    new.fit(newsale)

    print(new.predict(newsale[0:1]))