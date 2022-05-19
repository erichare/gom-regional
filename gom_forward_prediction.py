import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from copy import deepcopy
import time
import math
import tensorflow as tf
from tensorflow import keras
import concurrent.futures
import streamlit as st
from io import BytesIO
import pandas as pd
import st_holoview as sth
import json
import holoviews as hv
import pickle


##################################################################################################
class NN:
	def __init__(self, model, out_property_name, out_property_unit):
		self.model = keras.models.load_model(model, compile=False)
		self.out_property_name = out_property_name
		self.out_property_unit = out_property_unit

cdict = {'red': [(0.0, 0.0078, 0.0078),
                 (0.06249, 0.0078, 0.0078),
                 (0.0625, 0.0, 0.0),
                 (0.34375, 1.0, 1.0),
                 (0.46875, 1.0, 1.0),
                 (0.8125, 1.0, 1.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)],
        'green': [(0.0, 0.0078, 0.0078),
                 (0.06249, 0.0078, 0.0078),
                 (0.0625, 0.58, 0.58),
                 (0.34375, 1.0, 1.0),
                 (0.46875, 0.0, 0.0),
                 (0.8125, 0.0, 0.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)],
        'blue': [(0.0, 1.0, 1.0),
                 (0.06249, 1.0, 1.0),
                 (0.0625, 0.0, 0.0),
                 (0.34375, 0.0, 0.0),
                 (0.46875, 0.0, 0.0),
                 (0.8125, 0.0, 0.0),
                 (0.81251, 0.85, 0.85),
                 (1.0, 0.85, 0.85)]


        }

cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
path = 'data/'

mat_model = NN(path + 'trained_model_EzRo.h5', 'EasyRo', 'EzRo')
history_model = NN(path + 'nn_mat_history.h5', 'EasyRo', 'EzRo')
temp_model = NN(path + 'trained_model_Temperature.h5', 'Temperature', 'C')
transform_data = np.load(path + 'transform.npz')

affine = np.load(path + 'project_affine.npy')

d = np.load(path + 'mapstack.npz')
data_array = d['mapstack']


data_array[data_array<-9000] = np.nan

for i in range(5):
	thickness = data_array[i+1] - data_array[i]
	thickness[thickness < 1] = 10.0
	data_array[i+1] = data_array[i] + thickness

f = 'data/STS_ezRo.csv'
ro_sts = pd.read_csv(f, sep=';')

depths = data_array[:6]
lithos = data_array[6:11]


##################################################################################################
def transform(X, A, B, gradient, intercept):

	return (np.diag(1. / (B.dot(X) + gradient))).dot((A.dot(X) - intercept))


##################################################################################################
def group_transform(largeX, A, B, gradient, intercept):
	return [transform(X, A, B, gradient, intercept) for X in largeX]

##################################################################################################
def split_data_array(data_array, batches):
	nb_samples = data_array.shape[0]
	lim = int(nb_samples / batches)

	largeX_list = [data_array[i*lim:(i+1)*lim] for i in range(batches-1)]
	largeX_list.append(data_array[(batches-1)*lim:nb_samples])

	return largeX_list

##################################################################################################
def prepare_vector(transform_data, data_array):

	u = transform_data
	A = u['A']
	B = u['B']
	gradient = u['gradient']
	intercept = u['intercept']

	batches = 1
	data_array2 = deepcopy(data_array.T)

	if batches < 4:
		final = np.vstack([transform(a, A, B, gradient, intercept) for a in data_array2])
	else:
		largeX_list = split_data_array(data_array2, batches)

		executor = concurrent.futures.ProcessPoolExecutor(10)
		result = [executor.submit(group_transform, largeX, A, B, gradient, intercept) for largeX in largeX_list]
		concurrent.futures.wait(result)

		final = np.vstack([list(r.result()) for r in list(result)])
	print("FINAL SHAPE", final.shape)

	final[final < 0.] = 0.
	final[final > 1.] = 1.

	return final

##################################################################################################
def predict(input, model):

	batch_size = 300000
	prediction = model.predict(input, verbose=1, batch_size=batch_size)

	return prediction



##################################################################################################
def compute(input_vectors, mat_model, temp_model):

	
	temperature = predict(input_vectors, temp_model.model)
	maturity = predict(input_vectors, mat_model.model)

	return temperature, maturity

##################################################################################################
def compute_history(input_vectors, history_model):

	history = predict(input_vectors, history_model.model)

	return history

##################################################################################################
def get_predictions(data, variable):

	if len(data.shape) != 3:
		return "Found shape " + str(data.shape) + " Please provide a data array with shape (16, ny, nx)."

	else:
		ny = data.shape[1]
		nx = data.shape[2]
		array_to_compute = data.reshape((data.shape[0], nx*ny))
		input_vectors = prepare_vector(transform_data, data_array = array_to_compute)
		models = {'temperature' : temp_model.model, 'maturity': mat_model.model, 'maturity_history' : history_model.model}

		if variable in models:
			prediction = predict(input_vectors, models[variable])
			return prediction.transpose().reshape((prediction.shape[1], ny, nx))

		else:
			return "Unsupported variable type. Please use temperature, maturity or maturity_history"
	

##################################################################################################
def load_GOM_data(path):
	mat_model = NN(path + 'trained_model_EzRo.h5', 'EasyRo', 'EzRo')
	history_model = NN(path + 'nn_mat_history.h5', 'EasyRo', 'EzRo')
	temp_model = NN(path + 'trained_model_Temperature.h5', 'Temperature', 'C')
	transform_data = np.load(path + 'transform.npz')
	
	project_affine = np.load(path + 'project_affine.npy')

	d = np.load(path + 'mapstack.npz')
	data_array = d['mapstack']


	data_array[data_array<-9000] = np.nan

	for i in range(5):
		thickness = data_array[i+1] - data_array[i]
		thickness[thickness < 1] = 10.0
		data_array[i+1] = data_array[i] + thickness
	
	f = 'data/STS_ezRo.csv'
	ro_sts = pd.read_csv(f, sep=';')

	depths = data_array[:6]
	lithos = data_array[6:11]
	
	return transform_data, mat_model, temp_model, history_model, project_affine, data_array, ro_sts, depths, lithos



##################################################################################################
def st_ui():
	st.set_page_config(layout = "wide")

	# Load NN models and transform
	
	# transform_data, mat_model, temp_model, history_model, affine, data_array, ro_sts, depths, lithos = load_GOM_data(path)
	print(affine)
	dx = affine[6]
	dy = affine[7]
	nb_layers = 5
	ny = 492
	nx = 443


	start = time.time()

	layers_dict = {0: "Layer 1 - Plio-Pleistocene",
						1: "Layer 2 - Miocene",
						2: "Layer 3 - Paleogene",
						3: "Layer 4 - Late Cretaceous",
						4: "Layer 5 - Mid Jurassic to Mid Cretaceous"
						}
	layers_list = [layers_dict[i] for i in range(0,5)]


	ind_to_ages = np.load('data/ages_keys.npy')

	st.title('Regional Gulf Coast model')

	if 'mantle' not in st.session_state:
		st.session_state.mantle = -100

	if 'depth_uncertainty' not in st.session_state:
		st.session_state.depth_uncertainty = -100

	if 'crust_thickness' not in st.session_state:
		st.session_state.crust_thickness = -100

	if 'upper_crust_RHP' not in st.session_state:
		st.session_state.upper_crust_RHP = -100

	if 'temperature' not in st.session_state:
		st.session_state.temperature = 0

	if 'maturity' not in st.session_state:
		st.session_state.maturity = 0

	if 'history' not in st.session_state:
		st.session_state.history = 0

	if 'layer_select' not in st.session_state:
		st.session_state.layer_select = 0
	

	option = st.sidebar.selectbox("Layer selection", layers_list)

	time_event = 45

	property_list = ['Standard Thermal Stress', 'Temperature', 'Depth', 'Upper Crust RHP (uW/m3)', 'Crust Thickness (m)', 'Upper mantle thickness (m)']
	property = st.sidebar.selectbox("Property selection", property_list)

	if option == 'Layer 5 - Mid Jurassic to Mid Cretaceous' and property == 'Standard Thermal Stress':
		# age_select = st.sidebar.slider("Time section", -np.max(ind_to_ages[:,1]),0.0,0.0)
		age_select = st.sidebar.select_slider("Time selection", ind_to_ages[:,1], value = 0.0)
		# print(age_select)
		time_event = np.where(ind_to_ages[:,1] == age_select)[0]
		# print(time_event)
		# print(ind_to_ages)
		time_event = time_event

	upper_mantle_thick = st.sidebar.slider('Upper Mantle Thickness Uncertainty (%)', -20,20,0)
	upper_mantle_thick /= 100

	crust_thickness = st.sidebar.slider('Crust Thickness Uncertainty (%)', -20,20,0)
	crust_thickness /= 100

	upper_crust_RHP = st.sidebar.slider('Crust RHP Uncertainty (%)', -50,50,0)
	upper_crust_RHP /= 100


	depth_uncertainty = st.sidebar.slider('Depth Uncertainty (%). Default = 0%', -10,10,0)
	depth_uncertainty /= 100

	update_display = False

	array_to_compute = deepcopy(data_array)

	if upper_mantle_thick != st.session_state.mantle or \
		depth_uncertainty != st.session_state.depth_uncertainty or \
		crust_thickness != st.session_state.crust_thickness or \
		upper_crust_RHP != st.session_state.upper_crust_RHP: 

		array_to_compute[-2,:] *= (1 + upper_mantle_thick)
		array_to_compute[-4,:] *= (1 + upper_crust_RHP)
		array_to_compute[-5,:] *= (1 + crust_thickness)

		for i in range(1, nb_layers + 1):
			array_to_compute[i,:] *= (1 + depth_uncertainty)
		array_to_compute = array_to_compute.reshape((array_to_compute.shape[0], nx*ny))
		depths = data_array[:6]
		input_vectors = prepare_vector(transform_data, data_array = array_to_compute)
		temperature, maturity = compute(input_vectors, mat_model, temp_model)
		history = compute_history(input_vectors, history_model)
		print("HISTORY", history.shape)

			
		st.session_state.mantle = upper_mantle_thick
		st.session_state.depth_uncertainty = depth_uncertainty
		st.session_state.temperature = temperature
		st.session_state.maturity = maturity
		st.session_state.layer_select = option
		st.session_state.history = history
		st.session_state.crust_thickness = crust_thickness
		st.session_state.upper_crust_RHP = upper_crust_RHP

		update_display = True
		update_wells = True


	temperature = st.session_state.temperature
	maturity = st.session_state.maturity
	history = st.session_state.history
	for k, it in layers_dict.items():
		if option == it:
			index = k
	try:
		print(history[100000,44])
	except:
		pass

	print("Done", time.time() - start)

	# Get temperature and maturity grids
	new_temperature_lyr = deepcopy(temperature[:,index].reshape((ny, nx)))

	#Convert EasyRo to STS
	if time_event == 45:
		sts = np.interp(maturity[:,index]/100, ro_sts['ezRo'], ro_sts['sts'])
	else:
		sts = np.interp(history[:,time_event]/100, ro_sts['ezRo'], ro_sts['sts'])
	new_maturity_lyr = deepcopy(sts.reshape((ny, nx)))

	smoothed_mat =new_maturity_lyr
	smoothed_temp = new_temperature_lyr
	well_display = st.sidebar.checkbox('Display location of wells used for calibration', value=True)
	contours_display = st.sidebar.checkbox('Display contours', value=True)

	xp = np.linspace(0, nx+1, nx)
	yp = np.linspace(0, ny+1, ny)

	smoothed_mat = np.nan_to_num(smoothed_mat, nan=-9999)

	property_dict = {'Standard Thermal Stress' : smoothed_mat, 'Temperature' : smoothed_temp, 'Depth' : array_to_compute[index].reshape((ny, nx)),
					'Upper Crust RHP (uW/m3)' : array_to_compute[-4].reshape((ny, nx)), 
					'Crust Thickness (m)' : array_to_compute[-5].reshape((ny, nx)), 
					'Upper mantle thickness (m)' : array_to_compute[-2].reshape((ny, nx))}

	da = sth.create_xarray(property_dict[property], 
							xmin=affine[2],
							ymax = affine[5], 
							nx = nx,
							ny = ny,
							dx = dx,
							dy = dy)

	# if 'rxy' not in st.session_state:
	# 	st.session_state.rxy = None

	hv_plot = sth.create_hv_plot(da, well_display, property, contours_display)
	# st.session_state.rxy = rxy

	st.bokeh_chart(hv.render(hv_plot, backend='bokeh'))

if __name__ == "__main__":
	st_ui()
