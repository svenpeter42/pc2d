# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import functools
import traceback
import numpy

import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from threadpool import *
import time
import vigra
import os


class H5DSetWrapper:
	def __init__(self, dset, axis, dset_labels=None):
		self.dset = dset
		self.dset_labels = dset_labels
		self.axis = filter(lambda x: x in 'xyct', axis)

	@property
	def n_slices(self):
		if 't' in self.axis:
			return self.dset.shape[self.axis.find('t')]
		else:
			return 1

	@property
	def n_channels(self):
		if 'c' in self.axis:
			return self.dset.shape[self.axis.find('c')]
		else:
			return 1

	@property
	def n_x(self):
		if 'x' in self.axis:
			return self.dset.shape[self.axis.find('x')]
		else:
			return 1

	@property
	def n_y(self):
		if 'y' in self.axis:
			return self.dset.shape[self.axis.find('y')]
		else:
			return 1

	def __slice_h5(self, dset, slice_):
		for i, el in enumerate(slice_):
			if el != None: continue
			slice_[i] = (0, self.dset.shape[i])

		if len(slice_) == 2:
			return dset[slice_[0][0]:slice_[0][1],slice_[1][0]:slice_[1][1]]
		elif len(slice_) == 3:
			return dset[slice_[0][0]:slice_[0][1],slice_[1][0]:slice_[1][1],slice_[2][0]:slice_[2][1]]
		elif len(slice_) == 4:
			return dset[slice_[0][0]:slice_[0][1],slice_[1][0]:slice_[1][1],slice_[2][0]:slice_[2][1],slice_[3][0]:slice_[3][1]]

	def __get_block(self, dset, i_slice, x0, x1, y0, y1):
		tslice = (i_slice, i_slice+1)
		xslice = (x0, x1)
		yslice = (y0, y1)

		slicing = [None for _ in range(len(dset.shape))]

		if self.axis.find('t') >= 0: slicing[self.axis.find('t')] = tslice
		if self.axis.find('x') >= 0: slicing[self.axis.find('x')] = xslice
		if self.axis.find('y') >= 0: slicing[self.axis.find('y')] = yslice

		block = self.__slice_h5(dset, slicing).squeeze()

		if block.ndim == 2:
			block.shape += (1,)
		elif block.ndim == 3:
			pass
		else:
			raise ValueError("block.ndim == %d" % block.ndim)

		return block

	def get_block(self, i_slice, x0, x1, y0, y1):
		return self.__get_block(self.dset, i_slice, x0, x1, y0, y1)

	def get_label_block(self, i_slice, x0, x1, y0, y1):
		return self.__get_block(self.dset_labels, i_slice, x0, x1, y0, y1)

def getSlicing(begin, end):
    return [slice(b,e) for b,e in zip(begin,end)]

def getShape(begin, end):
	return (end[0] - begin[0], end[1] - begin[1])

def addHalo(shape, blockBegin, blockEnd, halo):

    withHaloBlockBegin = (
        max(blockBegin[0] - halo[0],0), 
        max(blockBegin[1] - halo[1],0)
    )

    withHaloBlockEnd = (
        min(blockEnd[0] + halo[0],shape[0]), 
        min(blockEnd[1] + halo[1],shape[1]) 
    )


    inBlockBegin = (
        blockBegin[0] -  withHaloBlockBegin[0],
        blockBegin[1] -  withHaloBlockBegin[1]
    )

    inBlockEnd = (
        inBlockBegin[0] +  (blockEnd[0] - blockBegin[0]),
        inBlockBegin[1] +  (blockEnd[1] - blockBegin[1])
    )

    return  withHaloBlockBegin, withHaloBlockEnd, inBlockBegin, inBlockEnd


def blockYielder(begin, end, blockShape):
    
    blockIndex = 0 
    for xBegin in range(begin[0], end[0], blockShape[0]):
        xEnd = min(xBegin + blockShape[0],end[0])
        for yBegin in range(begin[1], end[1], blockShape[1]):
            yEnd = min(yBegin + blockShape[1],end[1])
            yield blockIndex, (xBegin, yBegin), (xEnd, yEnd)
            blockIndex += 1

def forEachBlock(shape, blockShape, f, nWorker, roiBegin=None, roiEnd=None):
    if roiBegin is None:
        roiBegin = (0,0)

    if roiEnd is None:
        roiEnd = shape

    if nWorker <= 0:
    	nWorker = multiprocessing.cpu_count()

    if nWorker == 1:
        for blockIndex, blockBegin, blockEnd in blockYielder(roiBegin, roiEnd, blockShape):
            f(blockIndex=blockIndex, blockBegin=blockBegin, blockEnd=blockEnd)

    else:

        if False:
            futures = []
            with ThreadPoolExecutor(max_workers=nWorker) as executer:
                for blockIndex, blockBegin, blockEnd in blockYielder(roiBegin, roiEnd, blockShape):
                    executer.submit(f, blockIndex=blockIndex,blockBegin=blockBegin, blockEnd=blockEnd)

            for future in futures:
                e = future.exception()
                if e is not None:
                    raise e


        if True:
            pool = ThreadPool(nWorker)
            for blockIndex, blockBegin, blockEnd in blockYielder(roiBegin, roiEnd, blockShape):
                pool.add_task(f, blockIndex=blockIndex,blockBegin=blockBegin, blockEnd=blockEnd)
              
            pool.wait_completion()




def labelsBoundingBox(labels, blockBegin, blockEnd):
    whereLabels = numpy.array(numpy.where(labels!=0))

    inBlockBegin = numpy.min(whereLabels,axis=1)
    inBlockEnd   = numpy.max(whereLabels,axis=1) +1
    
    subBlockShape = [e-b for e,b in zip(inBlockEnd, inBlockBegin)]



    labelsBlock = labels[inBlockBegin[0]:inBlockEnd[0], 
                         inBlockBegin[1]:inBlockEnd[1]]

    globalBlockBegin = (
        blockBegin[0] + inBlockBegin[0], 
        blockBegin[1] + inBlockBegin[1]
    )

    globalBlockEnd = (
        blockBegin[0] + inBlockBegin[0]+subBlockShape[0], 
        blockBegin[1] + inBlockBegin[1]+subBlockShape[1]
    )

    whereLabels[0,:] -= inBlockBegin[0]
    whereLabels[1,:] -= inBlockBegin[1]

    return labelsBlock, globalBlockBegin, globalBlockEnd, whereLabels





class ProgressPrinter:
    def __init__(self, total, prefix='', suffix='', decimals=1, bar_length=100):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.bar_length = bar_length
        self(0)

    def __call__(self, iteration=None):
		# https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a
		# http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
		# https://creativecommons.org/licenses/by-sa/2.5/
	    """
	    Call in a loop to create terminal progress bar
	    @params:
	        iteration   - Required  : current iteration (Int)
	        total       - Required  : total iterations (Int)
	        prefix      - Optional  : prefix string (Str)
	        suffix      - Optional  : suffix string (Str)
	        decimals    - Optional  : positive number of decimals in percent complete (Int)
	        bar_length  - Optional  : character length of bar (Int)
	    """

	    if iteration == None:
	    	iteration = self.current + 1

	    self.current = iteration
	    str_format = "{0:." + str(self.decimals) + "f}"
	    percents = str_format.format(100 * (iteration / float(self.total)))
	    filled_length = int(round(self.bar_length * iteration / float(self.total)))
	    bar = '█' * filled_length + '-' * (self.bar_length - filled_length)

	    sys.stdout.write('\r%s |%s| %s%s %s' % (self.prefix, bar, percents, '%', self.suffix)),

	    if iteration == self.total:
	        sys.stdout.write('\n')
	    sys.stdout.flush()
