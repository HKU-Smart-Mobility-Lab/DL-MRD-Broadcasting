import numpy as np
import pandas as pd
from numpy import *
import pickle
import random
from io import StringIO
import os
import sys

driver_behaviour_model = pickle.load(open('./Driver_Behaviour.pickle', 'rb'))


def order_driver_info(radius, driver_info_filepath, order_info_filepath, output_filepath):
	driver_data = pd.read_csv(driver_info_filepath)
	order_data = pd.read_csv(order_info_filepath)

	num_order = order_data['order_id'].nunique()
	num_driver = driver_data['driver_id'].nunique()
	driver_data_temp = driver_data.copy()
	driver_loc_array = np.tile(driver_data_temp.values, (num_order, 1))
	order_data_temp = order_data.copy()
	request_array = np.repeat(order_data_temp.values, num_driver, axis=0)
	dis_array = distance_array(request_array[:, 1:3], driver_loc_array[:, 1:3])

	order_driver_pair = np.vstack(
		[request_array[:, 0], request_array[:, 4], request_array[:, 2], request_array[:, 1], driver_loc_array[:, 0],
		 driver_loc_array[:, 3], driver_loc_array[:, 2], driver_loc_array[:, 1], request_array[:, 3],
		 dis_array[:]]).T
	order_driver_info = pd.DataFrame(order_driver_pair.tolist(),
	                                 columns=['order_id', 'order_region', 'order_lat', 'order_lng', 'driver_id',
	                                          'driver_region', 'driver_lat', 'driver_lng', 'reward_units',
	                                          'pick_up_distance'])
	matched_result = order_broadcasting(order_driver_info, radius, output_filepath)
	return matched_result, 200


def driver_decision(distance, reward, lr_model):
	"""

    :param reward: numpyarray, price of order
    :param distance: numpyarray, distance between current order to all drivers
    :param numpyarray: n, price of order
    :return: pandas.DataFrame, the probability of drivers accept the order.
    """
	r_dis, c_dis = distance.shape
	temp_ = np.dstack((distance, reward)).reshape(-1, 2)
	result = lr_model.predict_proba(temp_).reshape(r_dis, c_dis, 2)
	result = np.delete(result, 0, axis=2)
	result = np.squeeze(result, axis=2)
	return result


def generate_random_num(length):
	if length < 1:
		res = 0
	else:
		res = random.randint(0, length)
	return res


def distance_array(coord_1, coord_2):
	"""
    :param coord_1: array of coordinate
    :type coord_1: numpy.array
    :param coord_2: array of coordinate
    :type coord_2: numpy.array
    :return: the array of manhattan distance of these two-point pair
    :rtype: numpy.array
    """
	coord_1 = coord_1.astype(float)
	coord_2 = coord_2.astype(float)
	coord_1_array = np.radians(coord_1)
	coord_2_array = np.radians(coord_2)
	dlon = np.abs(coord_2_array[:, 0] - coord_1_array[:, 0])
	dlat = np.abs(coord_2_array[:, 1] - coord_1_array[:, 1])
	r = 6371

	alat = np.sin(dlat / 2) ** 2
	clat = 2 * np.arctan2(alat ** 0.5, (1 - alat) ** 0.5)
	lat_dis = clat * r

	alon = np.sin(dlon / 2) ** 2
	clon = 2 * np.arctan2(alon ** 0.5, (1 - alon) ** 0.5)
	lon_dis = clon * r

	manhattan_dis = np.abs(lat_dis) + np.abs(lon_dis)

	return manhattan_dis


def order_broadcasting(order_driver_info, broadcasting_scale, output_filepath):
	"""

    :param order_driver_info: the information of drivers and orders
    :param broadcasting_scale: the radius of order broadcasting
    :return: matched driver order pair
    """

	# num of orders and drivers
	num_order = order_driver_info['order_id'].nunique()
	num_driver = order_driver_info['driver_id'].nunique()
	id_order = order_driver_info['order_id'].unique()
	id_driver = order_driver_info['driver_id'].unique()

	dis_array = np.array(order_driver_info['pick_up_distance'], dtype='float32').reshape(num_order, num_driver)
	distance_driver_order = dis_array.reshape(num_order, num_driver)
	driver_region_array = np.array(order_driver_info['driver_region'], dtype='float32').reshape(num_order, num_driver)
	order_region_array = np.array(order_driver_info['order_region'], dtype='float32').reshape(num_order, num_driver)
	driver_lat_array = np.array(order_driver_info['driver_lat'], dtype='float32').reshape(num_order, num_driver)
	driver_lng_array = np.array(order_driver_info['driver_lng'], dtype='float32').reshape(num_order, num_driver)
	order_lat_array = np.array(order_driver_info['order_lat'], dtype='float32').reshape(num_order, num_driver)
	order_lng_array = np.array(order_driver_info['order_lng'], dtype='float32').reshape(num_order, num_driver)

	price_array = np.array(order_driver_info['reward_units'], dtype='float32').reshape(num_order, num_driver)

	radius_array = np.full((num_order, num_driver), broadcasting_scale, dtype='float32')
	driver_decision_info = driver_decision(distance_driver_order, price_array, driver_behaviour_model)
	'''
    Choose Driver with probability
    '''
	for i in range(num_order):
		for j in range(num_driver):
			if distance_driver_order[i, j] > radius_array[i, j]:
				driver_decision_info[i, j] = 0  # delete drivers further than broadcasting_scale
	# match_state_array[i, j] = 2

	random.seed(10)
	temp_random = np.random.random((num_order, num_driver))
	driver_pick_flag = (driver_decision_info > temp_random) + 0
	driver_id_list = []
	order_id_list = []
	reward_list = []
	pick_up_distance_list = []
	index = 0
	for row in driver_pick_flag:
		temp_line = np.argwhere(row == 1)
		if len(temp_line) >= 1:
			temp_num = generate_random_num(len(temp_line) - 1)
			row[:] = 0
			row[temp_line[temp_num, 0]] = 1
			driver_pick_flag[index, :] = row
			driver_pick_flag[index + 1:, temp_line[temp_num, 0]] = 0

		index += 1

	matched_pair = np.argwhere(driver_pick_flag == 1)
	matched_dict = {}
	for item in matched_pair:
		matched_dict[id_order[item[0]]] = [order_region_array[item[0], item[1]], order_lat_array[item[0], item[1]],
		                                   order_lng_array[item[0], item[1]], id_driver[item[1]],
		                                   driver_region_array[item[0], item[1]], driver_lat_array[item[0], item[1]],
		                                   driver_lng_array[item[0], item[1]]]
		driver_id_list.append(id_driver[item[1]])
		order_id_list.append(id_order[item[0]])
		reward_list.append(price_array[item[0], item[1]])

		pick_up_distance_list.append(distance_driver_order[item[0], item[1]])
	result = []
	for item in id_order.tolist():
		if item in matched_dict:
			result.append(
				[item, matched_dict[item][0], matched_dict[item][1], matched_dict[item][2], matched_dict[item][3],
				 matched_dict[item][4], matched_dict[item][5], matched_dict[item][6], broadcasting_scale])

	result_columns = ['order_id', 'order_region', 'order_lat', 'order_lng', 'driver_id', 'driver_region', 'driver_lat',
	                  'driver_lng', 'radius']

	# 创建 DataFrame
	result_df = pd.DataFrame(result, columns=result_columns)
	result_df.to_csv(output_filepath)
	print(f"The matched result has been saved to {output_filepath}")
	return result_df.to_json(orient='records')


if __name__ == '__main__':
	radius = 5  # km
	driver_datapath = "./driver_info.csv"
	order_datapath = "./order_info.csv"
	output_filepath = "./matched_result.csv"
	order_driver_info(radius, driver_datapath, order_datapath,output_filepath)
