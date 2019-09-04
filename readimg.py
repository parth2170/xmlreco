import numpy as np 
import json
import ast
import array
import pickle
import gc
import os
import progressbar				

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

def image_to_dict(image_path):
	i = 1
	num = 0
	#prod_feat = {}
	ids = []
	pfs = []
	print("Reading Images")
	bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
	try:
		for image in readImageFeatures(image_path):
			bar.update(i)
			i += 1
			im, ft = image
			im = im.decode("utf-8")
			pfs.append(np.array(ft))
			if(i%500000) == 1:
				np.save('saved/pfeats'+str(num)+'.npy', pfs)
				num += 1
				i += 1
				del pfs
				pfs = []
				gc.collect()
	except EOFError:
		np.save('saved/pfeats.npy', pfs)
		print("File read")

	del pfs
	gc.collect()
	i = 0
	bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)

	try:
		for image in readImageFeatures(image_path):
			bar.update(i)
			i += 1
			im, ft = image
			im = im.decode("utf-8")
			ids.append(im)
			if(i%200000) == 1:
				num += 1
				i += 1
				gc.collect()
	except EOFError:
		np.save('saved/pids.npy', ids)
		print("File read")
	del ids
	gc.collect()
		


if __name__ == '__main__':
	image_to_dict('data/image_features')
	



