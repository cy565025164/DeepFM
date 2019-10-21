import os

# set the path-to-files
TRAIN_FILE = "./data/train.tmp"
VALID_FILE = None
TEST_FILE = "./data/tes"
feature_file = "all_feature.txt"

model_name = "DeepFM"

LABEL="falg_cate_1"

HAS_SCALE_BIN=False

chunk_size = 10

if model_name == "DeepFM":
    sub_dir = "./DeepFM_output"
    result_dir = "DeepFM_result"
if model_name == "FM":
    sub_dir = "./FM_output"
    result_dir = "FM_result"
if model_name == "DNN":
    sub_dir = "./DNN_output"
    result_dir = "DNN_result"

if not os.path.exists(sub_dir):
    os.makedirs(sub_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

NUM_SPLITS = 3
RANDOM_SEED = 2017

NUMERIC_COLS=["cpp_base_reg_days",
"buy_days_num",
"pv",
"visits",
"order_num",
"cart_num",
"sale_ord_1year_num",
"sale_ord_3month_num",
"sale_ord_2month_num",
"sale_ord_1week_num",
"click_times_15d",
"from_first_search_days",
"total_score",
"cps_arrive_num_15d",
"cps_use_num_15d",
"cps_use_radio_15d",
"cps_arrive_num_7d",
"cps_use_num_7d",
"cps_use_radio_7d",
"cps_arrive_num_3d",
"cps_use_num_3d",
"cps_use_radio_3d",
"cps_arrive_num_2d",
"cps_use_num_2d",
"cps_use_radio_2d",
"cps_arrive_num_1d",
"cps_use_num_1d",
"cps_use_radio_1d",
"from_last_click_cate_days",
"from_first_click_cate_days",
"sku_click_shoprel_num_cate_15d",
"skupage_dtl_click_num_cate_15d",
"skupage_click_num_cate_15d",
"comment_page_click_num_cate_15d",
"sku_click_shoprel_num_cate_7d",
"skupage_dtl_click_num_cate_7d",
"skupage_click_num_cate_7d",
"comment_page_click_num_cate_7d",
"sku_click_shoprel_num_cate_3d",
"skupage_dtl_click_num_cate_3d",
"skupage_click_num_cate_3d",
"comment_page_click_num_cate_3d",
"sku_click_shoprel_num_cate_2d",
"skupage_dtl_click_num_cate_2d",
"skupage_click_num_cate_2d",
"comment_page_click_num_cate_2d",
"sku_click_shoprel_num_cate_1d",
"skupage_dtl_click_num_cate_1d",
"skupage_click_num_cate_1d",
"comment_page_click_num_cate_1d",
"skupage_pv_cate_15d",
"skupage_pv_cate_7d",
"skupage_pv_cate_3d",
"skupage_pv_cate_2d",
"skupage_pv_cate_1d",
"from_last_search_cate_days",
"from_first_search_cate_days",
"search_times_cate_15d",
"search_times_cate_7d",
"search_times_cate_3d",
"search_times_cate_2d",
"search_times_cate_1d",
"from_last_cart_cate_days",
"from_first_cart_cate_days",
"cart_times_cate_15d",
"cart_times_cate_7d",
"cart_times_cate_3d",
"cart_times_cate_2d",
"cart_times_cate_1d",
"from_last_comment_days",
"from_first_comment_days",
"comment_times_1y",
"good_comment_times_1y",
"bad_comment_times_1y",
"comment_times_6m",
"good_comment_times_6m",
"bad_comment_times_6m",
"comment_times_3m",
"good_comment_times_3m",
"bad_comment_times_3m",
"comment_times_1m",
"good_comment_times_1m",
"bad_comment_times_1m",
"comment_times_15d",
"good_comment_times_15d",
"bad_comment_times_15d",
"comment_times_7d",
"good_comment_times_7d",
"bad_comment_times_7d",
"from_last_sale_ord_cate_days",
"from_first_sale_ord_cate_days",
"sale_ord_cate_num_1y",
"sale_ord_cate_num_6m",
"sale_ord_cate_num_3m",
"sale_ord_cate_num_1m",
"sale_ord_cate_num_15d",
"sale_ord_cate_num_7d",
"if_follow_cate_15d"]

CATEGORICAL_COLS=["cpp_base_ulevel",
"cpp_base_sex",
"cpp_base_age",
"cpp_base_marriage",
"cgp_cust_purchpower",
"cgp_cycl_lifecycle",
"cfv_sens_comment",
"cfv_sens_promotion",
"csf_saletm_first_ord_tm",
"csf_saletm_last_ord_tm",
"csf_saletm_last_login_tm",
"csf_sale_rebuy",
"csf_sale_rebuy_lasty",
"cvl_rfm_all_group",
"is_plus"]

IGNORE_COLS=['pin', 'falg_cate_1', 'falg_cate_5']
