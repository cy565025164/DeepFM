
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from sklearn.metrics import roc_auc_score

from sklearn.externals.joblib import load

import config
from DataReader import FeatureDictionary, DataParser
from DeepFM import DeepFM
import pickle
from tqdm import tqdm

from tensorflow.python.framework.graph_util import convert_variables_to_constants

import warnings
warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
conf = tf.ConfigProto()
conf.gpu_options.allow_growth=True
# conf.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=conf)

def read_d(file):
    lis = []
    for line in open(file, 'r'):
        line = line.strip().split(',')
        if len(line) == 0:
            continue
        lis.append(line[0].strip())
    return lis

def read_size(file):
    num = 0
    for _ in open(file, 'r'):
        num += 1
    return num

cols_name = read_d(config.feature_file)

def _load_data(path, cols_name):
    if not path:
        return None, None, None

    dfT = pd.read_csv(path, sep='\t', names=cols_name)

    cols = [c for c in dfT.columns if (not c in config.IGNORE_COLS)]

    X = dfT[cols].values
    y = dfT[config.LABEL].values

    return dfT, X, y


def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index+1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]

def batch_iter(Xi, Xv, y, batch_size):
    data_len = len(Xi)
    num_batch = int((data_len - 1) / batch_size) + 1

    Xi = np.array(Xi)
    Xv = np.array(Xv)
    y = np.array(y)

    indices = np.random.permutation(np.arange(data_len))
    xi_shuffle = Xi[indices]
    xv_shuffle = Xv[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield xi_shuffle[start_id:end_id], xv_shuffle[start_id:end_id], [[y_] for y_ in y_shuffle[start_id:end_id]]


# shuffle three lists simutaneously
def shuffle_in_unison_scary(a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)

def train_on_batch(model, sess, Xi, Xv, y, dropout_fm, dropout_deep, train_phase):
    feed_dict = {model.feat_index: Xi,
                 model.feat_value: Xv,
                 model.label: y,
                 model.dropout_keep_fm: dropout_fm,
                 model.dropout_keep_deep: dropout_deep,
                 model.train_phase: train_phase}
    out, loss, opt = sess.run((model.out, model.loss, model.optimizer), feed_dict=feed_dict)
    return loss

def _run_train(dfTrain, dfValid, dfm_params, SUB_DIR):
    save_path = os.path.join(SUB_DIR, 'model.ckpt')  # 最佳验证结果保存路径
    has_valid = dfValid is not None
    print ("feature dictionary...")
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfValid,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols=config.IGNORE_COLS,
                           scale=config.HAS_SCALE_BIN)
    with open(os.path.join(SUB_DIR, 'feat_dict.pickle'), 'wb') as fs:
        pickle.dump(fd.feat_dict, fs)


    # write feat_dict and scaler
    print("write feat_dict and scaler")
    fd_w = open(os.path.join(SUB_DIR, 'feat_dict.txt'), 'w')
    for k, v in fd.feat_dict.items():
        fd_w.write(str(k) + "\t")
        if isinstance(v, dict):
            for m, n in v.items():
                fd_w.write(str(m) + ":" + str(n) + "\t")
        if isinstance(v, int):
            fd_w.write(str(v))
        fd_w.write("\n")
    fd_w.close()

    sl_w = open(os.path.join(SUB_DIR, 'mean_scale.txt'), 'w')
    scale_ = fd.scale.scale_.tolist()
    mean_ = fd.scale.mean_.tolist()
    for i, j, k in zip(config.NUMERIC_COLS, mean_, scale_):
        sl_w.write(str(i) + "\t" + str(j) + "\t" + str(k) + "\n")
    sl_w.close()


    print ("data parser...")
    data_parser = DataParser(feat_dict=fd.feat_dict, scale=fd.scale)

    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    if has_valid:
        Xi_valid, Xv_valid, y_valid = data_parser.parse(df=dfValid, has_label=True)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    dfm = DeepFM(**dfm_params)
    valid_score = 0.0
    print ("start to train...")
    for epoch in range(dfm.epoch):
        t1 = time()
        # shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
        # total_batch = int((len(y_train) - 1) / dfm.batch_size) + 1
        # for i in range(total_batch):
        #     Xi_batch, Xv_batch, y_batch = get_batch(Xi_train, Xv_train, y_train, dfm.batch_size, i)
        #     train_on_batch(dfm, dfm.sess, Xi_batch, Xv_batch, y_batch, dfm.dropout_fm, dfm.dropout_deep, True)

        batch_train = batch_iter(Xi_train, Xv_train, y_train, dfm.batch_size)
        for xi_batch, xv_batch, y_batch in batch_train:
            train_on_batch(dfm, dfm.sess, xi_batch, xv_batch, y_batch, dfm.dropout_fm, dfm.dropout_deep, True)

        # evaluate training and validation datasets
        train_r = evaluate(dfm, dfm.sess, Xi_train, Xv_train, y_train)
        if has_valid:
            valid_r = evaluate(dfm, dfm.sess, Xi_valid, Xv_valid, y_valid)
            print("[%d] train-result=%.4f, valid-result=%.4f [%.1f s]"
                % (epoch + 1, train_r, valid_r, time() - t1))
            if valid_r > valid_score:
                valid_score = valid_r
                dfm.saver.save(sess=dfm.sess, save_path=save_path)
        else:
            print("[%d] train-result=%.4f [%.1f s]"
                % (epoch + 1, train_r, time() - t1))
    if not has_valid:
        dfm.saver.save(sess=dfm.sess, save_path=save_path)

def _predict(model, sess, Xi, Xv):
    # dummy y
    dummy_y = [1] * len(Xi)
    batch_index = 0
    Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, config.chunk_size, batch_index)
    y_pred = None
    model.batch_norm = 0
    while len(Xi_batch) > 0:
        num_batch = len(y_batch)
        feed_dict = {model.feat_index: Xi_batch,
                     model.feat_value: Xv_batch,
                     model.label: y_batch,
                     model.dropout_keep_fm: 1.0,
                     model.dropout_keep_deep: 1.0,
                     model.train_phase: False}
        batch_out = sess.run(model.out, feed_dict=feed_dict)
        if batch_index == 0:
            y_pred = np.reshape(batch_out, (num_batch,))
        else:
            y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

        batch_index += 1
        # print (batch_index)
        Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, config.chunk_size, batch_index)

    return y_pred

def evaluate(model, sess, Xi, Xv, y):
    y_pred = _predict(model, sess, Xi, Xv)
    return roc_auc_score(y, y_pred)

def _run_predict(dfTest, sub_dir):
    with open(os.path.join(sub_dir, 'feat_dict.pickle'), 'rb') as fs:
        feat_dict = pickle.load(fs)
    with open(os.path.join(sub_dir, 'params.pickle'), 'rb') as fs:
        params = pickle.load(fs)
    dfm = DeepFM(**params)
    scaler = load(os.path.join(sub_dir, 'scale.bin'))
    data_parser = DataParser(feat_dict=feat_dict,scale=scaler)

    print('start data parse...')
    Xi_test, Xv_test, y_test = data_parser.parse(df=dfTest, has_label=True)
    print ('start load model...')
    save_path = os.path.join(sub_dir, 'model.ckpt')
    dfm.saver.restore(sess=dfm.sess, save_path=save_path)  # 读取保存的模型

    y_test_meta = _predict(dfm, dfm.sess, Xi_test, Xv_test)

    return y_test_meta

# params
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": 1.0,
    "deep_layers": [32, 32],
    "dropout_deep": 0.5,
    "deep_layers_activation": tf.nn.relu,
    "epoch": 5,
    "batch_size": 10,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norm": 0,
    "batch_norm_decay": 0.997,
    "l2_reg": 0.01,
    "verbose": True,
    "random_seed": config.RANDOM_SEED
}

def train(model_name):
    # load data
    print ("load train and valid data...")
    dfTrain, X_train, y_train = _load_data(config.TRAIN_FILE, cols_name)
    dfValid, X_valid, y_valid = _load_data(config.VALID_FILE, cols_name)

    # ------------------ DeepFM Model ------------------
    if model_name == "DeepFM":
        print("DeepFM Model")
        _run_train(dfTrain, dfValid, dfm_params, config.sub_dir)

    # ------------------ FM Model ------------------
    if model_name == "FM":
        print("FM Model")
        fm_params = dfm_params.copy()
        fm_params["use_deep"] = False
        _run_train(dfTrain, dfValid, fm_params, config.sub_dir)

    # ------------------ DNN Model ------------------
    if model_name == "DNN":
        print("DNN Model")
        dnn_params = dfm_params.copy()
        dnn_params["use_fm"] = False
        _run_train(dfTrain, dfValid, dnn_params, config.sub_dir)


def predict_chunk(model_name, result_path):
    test_file_name = config.TEST_FILE.split('/')[-1]

    with open(os.path.join(config.sub_dir, 'feat_dict.pickle'), 'rb') as fs:
        feat_dict = pickle.load(fs)
    with open(os.path.join(config.sub_dir, 'params.pickle'), 'rb') as fs:
        params = pickle.load(fs)
    dfm = DeepFM(**params)
    scaler = load(os.path.join(config.sub_dir, 'scale.bin'))
    data_parser = DataParser(feat_dict=feat_dict, scale=scaler)

    print('start load model...')
    save_path = os.path.join(config.sub_dir, 'model.ckpt')
    dfm.saver.restore(sess=dfm.sess, save_path=save_path)  # 读取保存的模型

    print ("start read test file size...")
    num = read_size(config.TEST_FILE)
    print ("test file size: ", num)

    print("start predict...")
    pin, gr_test, testProb = [], [], []

    # for chunk in pd.read_csv(config.TEST_FILE, sep='\t', names=cols_name, chunksize=3):
    reader = pd.read_csv(config.TEST_FILE, sep='\t', names=cols_name, iterator=True)
    with tqdm(range(int((num-1)/config.chunk_size) + 1)) as t:
        for _ in t:
            chunk = reader.get_chunk(config.chunk_size)
            Xi_test, Xv_test, y_test = data_parser.parse(df=chunk, has_label=True)
            y_test_meta = _predict(dfm, dfm.sess, Xi_test, Xv_test)
            testProb.extend(y_test_meta)
            pin.extend(chunk["pin"].values.tolist())
            gr_test.extend(chunk[config.LABEL].values.tolist())
    print("write result...")
    _make_submission(pin, gr_test, testProb, result_path+"/"+model_name+"_"+test_file_name+"_result"+"_"+config.LABEL)

def predict(model_name, result_path):
    test_file_name = config.TEST_FILE.split('/')[-1]

    # load data
    print ("load test data...")
    dfTest, X_test, y_test = _load_data(config.TEST_FILE, cols_name)
    print ("start predict...")
    testProb = _run_predict(dfTest, config.sub_dir)
    print ("write result...")
    _make_submission(dfTest["pin"], y_test, testProb, result_path+"/"+model_name+"_"+test_file_name+"_result"+"_"+config.LABEL)

def _make_submission(ids, grd, testProb, filename="submission.csv"):
    print (len(ids), len(grd), len(testProb))
    pd.DataFrame({"pin": ids, "y_test":grd, "score": testProb}).to_csv(filename, index=False, float_format="%.5f", sep="\t", header=False)


def freeze_graph(input_checkpoint, output_graph):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    output_node_names = "my_output"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        for node in input_graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']

        output_graph_def = convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=input_graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

if __name__ == "__main__":
    train(config.model_name)
    freeze_graph(os.path.join(config.sub_dir, "model.ckpt"), os.path.join(config.sub_dir, "model.pb"))
    # predict_chunk(config.model_name, config.result_dir)
