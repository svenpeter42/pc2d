from __future__ import print_function
from __future__ import division

import sys
import os
import json
import h5py as h5
import numpy as np
import vigra

import tools
import features
import threading
import multiprocessing

def extract_train_data(settings):
    featuresList = []
    labelsList = []
    lock = threading.Lock()

    for name in settings["trainingData"]:
        raw_fname, raw_dset, raw_axis = settings["trainingData"][name]["raw"]
        labels_fname, labels_dset = settings["trainingData"][name]["labels"]

        with h5.File(raw_fname, "r") as fraw, h5.File(labels_fname, "r") as flabels:
            dset_raw = fraw[raw_dset]
            dset_labels = flabels[labels_dset]

            dset_wrapper = tools.H5DSetWrapper(dset_raw, raw_axis, dset_labels)

            shape = (dset_wrapper.n_x, dset_wrapper.n_y, dset_wrapper.n_channels)

            feats = features.ConvolutionFeatures(settings["setup"]["features"]["sigmas"], settings["setup"]["features"]["channels"])


            p = tools.ProgressPrinter(dset_wrapper.n_slices, suffix="Extracting training data")
            for i_slice in range(dset_wrapper.n_slices):
                def block_worker(blockIndex, blockBegin, blockEnd):
                    x0, y0 = blockBegin
                    x1, y1 = blockEnd

                    labels = dset_wrapper.get_label_block(i_slice, x0, x1, y0, y1)
                    if not labels.any(): return

                    haloBlockBegin, haloBlockEnd, inBlockBegin, inBlockEnd = tools.addHalo(shape, blockBegin, blockEnd, feats.halo)
                    hx0, hy0 = haloBlockBegin
                    hx1, hy1 = haloBlockEnd
                    raw = dset_wrapper.get_block(i_slice, hx0, hx1, hy0, hy1)

                    slicing = tools.getSlicing(inBlockBegin, inBlockEnd)

                    fvec_shape = tools.getShape(blockBegin, blockEnd) + (feats.n_features,)
                    raw_fvec = np.zeros(fvec_shape, dtype=np.float32)
                    feats(raw, slicing, raw_fvec)

                    slicing = np.where(labels != 0)

                    labels = labels[slicing].astype(np.uint32)
                    raw_fvec = raw_fvec[slicing[:2]]

                    with lock:
                        featuresList.append(raw_fvec)
                        labelsList.append(labels)

                p(i_slice)
                tools.forEachBlock(shape, tuple(settings["setup"]["blockShape"]), block_worker, settings["setup"]["n_jobs"])

            p(dset_wrapper.n_slices)

    feats = np.concatenate(featuresList, axis=0)
    labels = np.concatenate(labelsList,  axis=0)

    print("Training label distribution:", np.bincount(labels))

    return feats, labels


def train_rf(settings):
    print("Training Random Forest")
    with h5.File(settings["setup"]["cache"], "r") as f:
        feats = f["features"][:]
        labels = f["labels"][:]

    n_jobs = settings["setup"]["n_jobs"]
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count()

    rf = vigra.learning.RandomForest3(feats, labels, treeCount=settings["setup"]["n_trees"], n_threads=n_jobs)
    rf.writeHDF5(settings["setup"]["classifier"].encode("utf-8"), "/")

def train(settings):
    feats, labels = extract_train_data(settings)

    with h5.File(settings["setup"]["cache"], "w") as f:
        f["features"] = feats
        f["labels"] = labels

    train_rf(settings)

    print("Done training!")

def predict(settings):
    print("Loading Random Forest")

    rf = vigra.learning.RandomForest3(settings["setup"]["classifier"].encode("utf-8"), "/")

    lock = threading.Lock()

    for preddata in settings["predictionData"]:
        with h5.File(preddata["filename"], "r") as fraw, h5.File(preddata["output"]["filename"], "w") as fpred:
            dset_raw = fraw[preddata["dataset"]]

            dset_wrapper = tools.H5DSetWrapper(dset_raw, preddata["axis"])


            if settings["setup"]["predict_proba"]:
                dset_pred = fpred.create_dataset(preddata["output"]["dataset"], (dset_wrapper.n_slices, dset_wrapper.n_x, dset_wrapper.n_y, rf.labelCount()), chunks=True)
            else:
                dset_pred = fpred.create_dataset(preddata["output"]["dataset"], (dset_wrapper.n_slices, dset_wrapper.n_x, dset_wrapper.n_y, 1), chunks=True)

            shape = (dset_wrapper.n_x, dset_wrapper.n_y, dset_wrapper.n_channels)
            feats = features.ConvolutionFeatures(settings["setup"]["features"]["sigmas"], settings["setup"]["features"]["channels"])

            n_blocks = 0
            for _ in tools.blockYielder((0,0), shape, tuple(settings["setup"]["blockShape"])):
                n_blocks += 1

            p = tools.ProgressPrinter(dset_wrapper.n_slices * n_blocks, suffix="Predicting")
            for i_slice in range(dset_wrapper.n_slices):
                p(i_slice * n_blocks)

                def pred_worker(blockIndex, blockBegin, blockEnd):
                    haloBlockBegin, haloBlockEnd, inBlockBegin, inBlockEnd = tools.addHalo(shape, blockBegin, blockEnd, feats.halo)

                    x0, y0 = blockBegin
                    x1, y1 = blockEnd
                    hx0, hy0 = haloBlockBegin
                    hx1, hy1 = haloBlockEnd
                    raw = dset_wrapper.get_block(i_slice, hx0, hx1, hy0, hy1)

                    slicing = tools.getSlicing(inBlockBegin, inBlockEnd)

                    fvec_shape = tools.getShape(blockBegin, blockEnd) + (feats.n_features,)
                    raw_fvec = np.zeros(fvec_shape, dtype=np.float32)
                    feats(raw, slicing, raw_fvec)

                    raw_fvec = raw_fvec.reshape((-1, feats.n_features))

                    if settings["setup"]["predict_proba"]:
                        pred = rf.predictProbabilities(raw_fvec)
                    else:
                        pred = rf.predictLabels(raw_fvec)

                    with lock:
                        dset_pred[i_slice, x0:x1, y0:y1, :] = pred.reshape((x1-x0, y1-y0, -1))
                        p()


                tools.forEachBlock(shape, tuple(settings["setup"]["blockShape"]), pred_worker, settings["setup"]["n_jobs"])
            p(dset_wrapper.n_slices)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Invalid command line arguments.")

    with open(sys.argv[2], "r") as f:
        settings = json.load(f)

    if sys.argv[1] == "train": train(settings)
    elif sys.argv[1] == "predict": predict(settings)
    else: raise ValueError("Unsupported mode: %s" % sys.argv[1]) 