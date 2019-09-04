import os
import datetime
import random
import pickle
import json
from joblib import Parallel, delayed
import multiprocessing
import progressbar
from tqdm import tqdm
import gc
import numpy as np 


def read_data(reviews, ispickle, min_rating):

	user_prod_dict = {}
	prod_user_dict = {}
	bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
	print("\nOpening reviews file")
	i = 0
	if ispickle:
		with open(reviews, 'rb') as file:
			while(True):
				try:
					jline = pickle.load(file)
					dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
					if (dt['reviewerID'] not in user_prod_dict) and (dt['overall'] >= min_rating):
						user_prod_dict[dt['reviewerID']] = []
					if (dt['asin'] not in prod_user_dict) and (dt['overall'] >= min_rating):
						prod_user_dict[dt['asin']] = []
					if dt['overall'] > min_rating:	#Construct user product link only if overall rating is > min_rating
						user_prod_dict[dt['reviewerID']].append(dt['asin'])
						prod_user_dict[dt['asin']].append(dt['reviewerID'])
				except EOFError:
					break
	else:
		with open(reviews, 'r') as file:
			for line in file:
				bar.update(i)
				jline = json.loads(line)
				i+=1
				dt = dict((k, jline[k]) for k in ('reviewerID', 'asin', 'overall'))
				if dt['overall'] > min_rating:
					try:
						user_prod_dict[dt['reviewerID']].append(dt['asin'])
					except KeyError:
						user_prod_dict[dt['reviewerID']] = []
						user_prod_dict[dt['reviewerID']].append(dt['asin'])
					try:
						prod_user_dict[dt['asin']].append(dt['reviewerID'])
					except KeyError:
						prod_user_dict[dt['asin']] = []
						prod_user_dict[dt['asin']].append(dt['reviewerID'])
	print("\nData read")
	return user_prod_dict, prod_user_dict


if __name__ == '__main__':
	reviews = "data/reviews_Clothing_Shoes_and_Jewelry.json"
	user_prod_dict, prod_user_dict = read_data(reviews = reviews, ispickle = False, min_rating = 0)	
	with open('saved/user_prod_dict.pickle', 'wb') as file:
			pickle.dump(user_prod_dict, file)
	with open('saved/prod_user_dict.pickle', 'wb') as file:
			pickle.dump(prod_user_dict, file)