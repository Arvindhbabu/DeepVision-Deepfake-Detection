#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
from src.config import SEQS_DIR

def _bytes_feature(tf, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def pack_tfrecord(seq_path, out_path):
    try:
        import tensorflow as tf
    except Exception as e:
        raise ImportError(f"TensorFlow is required for TFRecord generation: {e}")
    seq = np.load(seq_path)
    raw = seq.tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
        'video_id': _bytes_feature(tf, os.path.basename(seq_path).encode('utf-8')),
        'seq': _bytes_feature(tf, raw)
    }))
    with tf.io.TFRecordWriter(out_path) as writer:
        writer.write(example.SerializeToString())

if __name__ == "__main__":
    for ds in ["ffpp", "celebdf"]:
        files = glob.glob(os.path.join(SEQS_DIR, ds, "*.npy"))
        out_dir = os.path.join(SEQS_DIR, "tfrecords", ds)
        os.makedirs(out_dir, exist_ok=True)
        for f in tqdm(files):
            out_file = os.path.join(out_dir, Path(f).stem + ".tfrecord")
            pack_tfrecord(f, out_file)


