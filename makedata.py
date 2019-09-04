import os
import gc
import json
import pickle
import numpy as np 
from tqdm import tqdm
from scipy.spatial.distance import pdist,squareform

def saver(data, num, ids):
	print('\nSaving dataset')
	with open('dataset/data'+str(num)+'.pickle', 'wb') as file:
		pickle.dump(data, file)
	pfs = []
	idsub = []
	for prod in data:
		idsub.append(prod)
		pfs.append(get_pf(prod, ids))
	np.save('dataset/pfs'+str(num)+'.npy', pfs)
	np.save('dataset/ids'+str(num)+'.npy', idsub)


def get_pf(pid, ids):
	index = list(ids).index(pid)
	pfs = np.load('saved/pfeats0.npy')
	l = len(pfs)
	if index > l:
		del pfs
		gc.collet()
		index -= l
		pfs = np.load('saved/pfeats1.npy')
		l = len(pfs)
		if index > l:
			del pfs
			gc.collet()
			index -= l
			pfs = np.load('saved/pfeats.npy')
			return pfs[index]
		else:
			return pfs[index]
	else:
		return pfs[index]


def makedata(ids, prod_user_dict):
	udata = []
	data = {}
	num = 0
	for prod in tqdm(prod_user_dict):
		udata.extend(prod_user_dict[prod])
		data[prod] = prod_user_dict[prod]
		if len(udata) >= 10000:
			saver(data, num)
			num += 1
			data = {}
			udata = []


def user_feat(reviews, ids, user_prod_dict):
	user_f = {}
	user_rate_dict = {}
	print("\nOpening reviews file")
	with open(reviews, 'r') as file:
		for line in file:
			jline = json.loads(line)
			dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
			try:
				user_rate_dict[dt['reviewerID']].append(dt['overall'])
			except KeyError:
				user_rate_dict[dt['reviewerID']] = []
				user_rate_dict[dt['reviewerID']].append(dt['overall'])
	print('Making user features')
	for user in tqdm(user_prod_dict):
		uf = []
		uid = []
		ufid = []
		for i in range(len(user_prod_dict[user])):
			pf = get_pf(user_prod_dict[user][i], ids)
			uf.append(pf*user_rate_dict[i])
		uf = np.sum(uf)/np.sum(user_rate_dict[user])
		user_f[user] = uf
		uid.append(user)
		ufid.append(uf)
	with open('saved/user_feat.pickle', 'wb') as file:
		pickle.dump(user_f, file)

	print('Making similarity matrix')
	sim_mat = 1 - np.array(squareform(pdist(ufid, metric='cosine')))
	del ufid
	gc.collect()
	np.save('saved/usim_mat.npy', sim_mat)
	np.save('saved/uids.npy', uids)



if __name__ == '__main__':
	print('Loading Dicts')
	ids = np.load('saved/pids.npy')
	with open('saved/user_prod_dict.pickle', 'rb') as file:
			up = pickle.load(file)
	with open('saved/prod_user_dict.pickle', 'rb') as file:
			pu = pickle.load(file)
	print('Making User features')
	user_feat('data/reviews_Clothing_Shoes_and_Jewelry.json', ids, up)
	print('Making data')
	makedata(up, pu)