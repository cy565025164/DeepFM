import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.externals.joblib import load

import config
from DataReader import DataParser
from DeepFM import DeepFM
import pickle
from tqdm import tqdm

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

def _predict(model, sess, Xi, Xv):
    # dummy y
    dummy_y = [1] * len(Xi)
    batch_index = 0
    Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, model.batch_size, batch_index)
    y_pred = None
    while len(Xi_batch) > 0:
        num_batch = len(y_batch)
        feed_dict = {model.feat_index: Xi_batch,
                     model.feat_value: Xv_batch,
                     model.label: y_batch,
                     model.dropout_keep_fm: 1.0,
                     model.dropout_keep_deep: 1.0,
                     model.train_phase: False}
        try:
            batch_out = sess.run(model.out, feed_dict=feed_dict)

        except:
            batch_out = np.array([[0.0]*num_batch])
        if batch_index == 0:
            y_pred = np.reshape(batch_out, (num_batch,))
        else:
            y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

        batch_index += 1
        # print (batch_index)
        Xi_batch, Xv_batch, y_batch = get_batch(Xi, Xv, dummy_y, model.batch_size, batch_index)

    return y_pred

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

    num = read_size(config.TEST_FILE)
    print("test file size: ", num)

    pin, gr_test, testProb = [], [], []

    # for chunk in pd.read_csv(config.TEST_FILE, sep='\t', names=cols_name, chunksize=3):
    reader = pd.read_csv(config.TEST_FILE, sep='\t', names=cols_name, iterator=True)
    with tqdm(range(int((num - 1) / config.chunk_size) + 1)) as t:
        for _ in t:
            chunk = reader.get_chunk(config.chunk_size)
            Xi_test, Xv_test, y_test = data_parser.parse(df=chunk, has_label=True)
            y_test_meta = _predict(dfm, dfm.sess, Xi_test, Xv_test)
            testProb.extend(y_test_meta)
            pin.extend(chunk["pin"].values.tolist())
            gr_test.extend(chunk[config.LABEL].values.tolist())
    print("write result...")
    _make_submission(pin, gr_test, testProb,
                     result_path + "/" + model_name + "_" + test_file_name + "_result" + "_" + config.LABEL)

def _make_submission(ids, grd, testProb, filename="submission.csv"):
    print(len(ids), len(grd), len(testProb))
    pd.DataFrame({"pin": ids, "y_test": grd, "score": testProb}).to_csv(filename, index=False, float_format="%.5f",
                                                                        sep="\t", header=False)

if __name__ == "__main__":
    config.HAS_SCALE_BIN=True
    config.LABEL="falg_cate_1"
    predict_chunk(config.model_name, config.result_dir)
