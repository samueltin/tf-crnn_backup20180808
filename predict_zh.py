import argparse
import os, time
import csv
import numpy as np
try:
    import better_exceptions
except ImportError:
    pass
from tqdm import trange
import tensorflow as tf
from src.model import crnn_fn
from src.data_handler import data_loader
from src.data_handler import preprocess_image_for_prediction
from src.read_dict import dict_as_str

from src.config import Params, Alphabet, import_params_from_json
import cv2
parameters = Params(train_batch_size=128,
eval_batch_size=128,
learning_rate=1e-3, # 1e-3 recommended
learning_decay_rate=0.95,
learning_decay_steps=5000,
evaluate_every_epoch=5,
save_interval=5e3,
input_shape=(32, 304),
optimizer='adam',
digits_only=True,
alphabet=dict_as_str(),
alphabet_decoding='same',
csv_delimiter='\t',
csv_files_eval='/Users/samueltin/Projects/jb/crnn_tf_data/Test/sample.csv',
csv_files_train='/Users/samueltin/Projects/jb/crnn_tf_data/Train/sample.csv',
output_model_dir='./estimator/',
n_epochs=1,
gpu=''
)
model_params = {
'Params': parameters,
}

parameters.export_experiment_params()

os.environ['CUDA_VISIBLE_DEVICES'] = parameters.gpu
config_sess = tf.ConfigProto()
config_sess.gpu_options.per_process_gpu_memory_fraction = 0.8
config_sess.gpu_options.allow_growth = True

est_config = tf.estimator.RunConfig()
est_config.replace(keep_checkpoint_max=10,
save_checkpoints_steps=parameters.save_interval,
session_config=config_sess,
save_checkpoints_secs=None,
save_summary_steps=1000,
model_dir=parameters.output_model_dir)

estimator = tf.estimator.Estimator(model_fn=crnn_fn,
params=model_params,
model_dir=parameters.output_model_dir,
config=est_config
)

predictResults=estimator.predict(input_fn=data_loader(csv_filename='/Users/samueltin/Projects/sf/sf-image-generator/output/Test/sample.csv',
                                            params=parameters,
                                            batch_size=1,
                                            num_epochs=1,
                                            data_augmentation=False,
                                            image_summaries=False
                                            ))
is_vis=False

ans_dict = {}
ans_file = open('/Users/samueltin/Projects/sf/sf-image-generator/output/Test/sample.csv', mode='r', encoding='utf-8')
content = ans_file.read()
ans_file.close()
lines = content.split('\n')
for line in lines:
    items = line.split('\t')
    if len(items)>1:
        chats = items[1].split('{')
        label = ''.join(chats)
        key = items[0]
        ans_dict[key]=label

total=1
correct=0



for i, prediction in enumerate(predictResults):
    words_in_bytes=prediction["words"]
    words = words_in_bytes.decode('utf-8')
    filename=prediction["filenames"].decode("utf-8")
    if is_vis:
        image = cv2.imread(filename)
        cv2.imshow('Test image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Label={}, Predict={}, File={}, total={}, correct={}'.format(ans_dict[filename], words, filename, total,
                                                                           correct))
    if words != ans_dict[filename]:
        print('Label={}, Predict={}, File={}, total={}, correct={}'.format(ans_dict[filename], words, filename, total, correct))
    else:
        correct=correct+1
    total=total+1

acc = int(correct / total * 100)

print("Accurarcy={}".format(str(acc)))