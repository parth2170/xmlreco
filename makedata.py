import os
import gc
import json
import pickle
import numpy as np 
from tqdm import tqdm
from scipy.spatial.distance import pdist,squareform

def user_feat(user_prod_dict, cat, feats, user_rate_dict):
	user_f = {}
	print('Making user features')
	gc.collect()
	for user in tqdm(user_prod_dict):
		uf = []
		uid = []
		ufid = []
		for i in range(len(user_prod_dict[user])):
			pf = feats[user_prod_dict[user][i]]
			uf.append(pf*user_rate_dict[i])
		uf = np.sum(uf)/np.sum(user_rate_dict[user])
		user_f[user] = uf
		uid.append(user)
		ufid.append(uf)
	with open('dataset/'+cat+'user_feat.pickle', 'wb') as file:
		pickle.dump(user_f, file)

	print('Making similarity matrix')
	sim_mat = 1 - np.array(squareform(pdist(ufid, metric='cosine')))
	del ufid
	gc.collect()
	np.save('dataset/'+cat+'usim_mat.npy', sim_mat)
	np.save('dataset/'+cat+'uids.npy', uids)


def reverse_dict(D):
	rD = {}
	for i in tqdm(D):
		for j in D[i]:
			try:
				rD[j].append(i)
			except KeyError:
				rD[j] = []
				rD[j].append(i)
	return rD

if __name__ == '__main__':
	print('Loading Dicts')
	with open('saved/prod_user_dict.pickle', 'rb') as file:
		pu = pickle.load(file)
	with open('saved/map_dict.pickle', 'rb') as file:
		mp = pickle.load(file)
	master_cats = ['Baby', 'Boots', 'Boys', 'Girls'] # 'Jewelry', 'Men', 'Novelty', 'Costumes', 'Shoes', 'Accessories', 'Women']
	user_rate_dict = {}
	print("\nOpening reviews file")
	with open('data/reviews_Clothing_Shoes_and_Jewelry.json', 'r') as (file):
		for line in tqdm(file):
			jline = json.loads(line)
			dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
			try:
				user_rate_dict[dt['reviewerID']].append(dt['overall'])
			except KeyError:
				user_rate_dict[dt['reviewerID']] = []
				user_rate_dict[dt['reviewerID']].append(dt['overall'])
	for cat in master_cats:
		with open('saved/pfeats_'+str(cat)+'.pickle', 'rb') as file:
			feats = pickle.load(file)
		temp = {}
		for prod in pu:
			if cat in mp[prod]:
				temp[prod] = pu[prod]
		user_prod_dict = reverse_dict(temp)
		with open('dataset/data_'+str(cat)+'.pickle', 'wb') as file:
			pickle.dump(temp, file)
		user_feat(user_prod_dict, cat, feats, user_rate_dict)
