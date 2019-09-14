import numpy as np 
import json
import ast
import array
import pickle
import gc
import os
from tqdm import tqdm

def read_meta(meta_data_path):
	unrelated = []
	prod_cats = {}
	master_cats = ['Baby', 'Boots', 'Boys', 'Girls', 'Jewelry', 'Men', 'Novelty', 'Costumes', 'Shoes', 'Accessories', 'Women']
	print("Reading Meta Data")
	with open(meta_data_path, 'r') as file:
		count = 1
		for line in tqdm(file):
			jline = ast.literal_eval(line)
			try:
				dt = dict((k, jline[k]) for k in ('asin', 'categories'))
				prod_id = dt['asin']
				cats = list(set([j for i in dt['categories'] for j in i]))
				flag = 0
				temp = []
				for cat in cats:
					if cat in master_cats:
						flag = 1
						if cat not in prod_cats:
							prod_cats[cat] = []
						prod_cats[cat].append(prod_id)
					else:
						temp.append(cat)
				for cat in temp:
					t = cat.split()
					for q in t:
						if q in master_cats:
							flag = 1
							if q not in prod_cats:
								prod_cats[q] = []
							prod_cats[q].append(prod_id)
				if flag == 0:
					print()
			except KeyError as error:
				pass
	for cat in prod_cats:
		prod_cats[cat] = list(set(prod_cats[cat]))
	prod_cats['Novelty Costumes'] = list(set(prod_cats['Novelty'] + prod_cats['Costumes']))
	prod_cats['Shoes and Accessories'] = list(set(prod_cats['Shoes'] + prod_cats['Accessories']))
	prod_cats.pop('Novelty')
	prod_cats.pop('Costumes')
	prod_cats.pop('Shoes')
	prod_cats.pop('Accessories')
	return prod_cats


def map(prod_cats):
	map_dict = {}
	for cat in prod_cats:
		for prod in prod_cats[cat]:
			try:
				map_dict[prod].append(cat)
			except:
				map_dict[prod] = [cat]
	return map_dict


def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    asin = f.read(10)
    if asin == '': break
    a = array.array('f')
    a.fromfile(f, 4096)
    yield asin, a.tolist()

def image_to_dict(image_path, map_dict, cat):
	prod_feat = {}
	try:
		for image in tqdm(readImageFeatures(image_path)):
			im, ft = image
			im = im.decode("utf-8")
			if cat in map_dict[im]:
				prod_feat[im] = ft
			del ft
	except EOFError:
		with open('saved/pfeats_'+str(cat)+'.pickle', 'wb') as file:
					pickle.dump(prod_feat, file)
		del prod_feat
		gc.collect()
		print("File read")		


if __name__ == '__main__':
	# cat_dict = read_meta('data/meta_Clothing_Shoes_and_Jewelry.json')
	# map_dict = map(cat_dict)
	# with open('saved/map_dict.pickle', 'wb') as file:
	# 	pickle.dump(map_dict, file)
	with open('saved/map_dict.pickle', 'rb') as file:
	 	map_dict = pickle.load(file)
	master_cats = ['Shoes and Accessories', 'Women']
	for cat in master_cats:
		print(cat)
		image_to_dict('data/image_features', map_dict, cat)
		gc.collect()

	



