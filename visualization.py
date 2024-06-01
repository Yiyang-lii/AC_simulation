import numpy as np
import numba as nb
from functions.Particles import Particles
from functions.DataProcesser import DataProcesser
from functions.Environment import Environment
from functions.Simulators import Simulators
import gc
import datetime
import matplotlib.pyplot as plt
start_time=datetime.datetime.today()
print('start_time=',start_time)

nthreads = 20
nb.set_num_threads(nthreads)

temperture_sigma=5
density_sigma=5
#'''
#set environment
filename='0p2heat_n2000_room5000_dT60'
filepaths=f'data/{filename}_t_0_100_0p1'


fns=DataProcesser.load_files(filepaths,filename) 

for i in [1, 500 ,1000]:
    i="{:05}".format(i)
    print(f't{i}')
    print(f'{filepaths}/{filename}_t{i}.bin')
    particles=DataProcesser.data_input(f'{filepaths}/{filename}_t{i}.bin')  
    DataProcesser.plot_velocity_distribution(temperature=particles.count_average_T(),mass=particles.mass,vel=particles.vel,save=True,filepath=filename,filename=f'velocity_distribution_{filename}_t{i}')
    DataProcesser.plot_gas_number_density(particles, resolution=100,sigma=density_sigma,fig_save=True,filename=f'gas_number_density_{filename}_t{i}',filepath=filename)
    DataProcesser.plot_gas_temperature(particles, resolution=100,vmin=270,vmax=330,sigma=temperture_sigma,fig_save=True,filename=f'gas_temperature_{filename}_t{i}',filepath=filename)
    print(f't{i}_average temperature = {particles.count_average_T()}')

DataProcesser.plot_temperature_versus_time(fns,save=True,filename=f'temperature_versus_time_{filename}_t{i}',filepath=filename)

#output movie  

#WARNING:output_particles_movie can almost see nothing when particles_number is more than 100000
DataProcesser.output_particles_movie(fns, filename=f'{filename}.mp4', fps=30,filepath=filename) #plot particles

#WARNING: The upper and lower bounds of the colorbar need to adjust manually in DataProcesser.py!!!
DataProcesser.output_movie(fns, resolution=100, sigma=temperture_sigma, filename=f"temperature_{filename}_sigma{temperture_sigma}.mp4", fps=10, plot_func="plot_gas_temperature",filepath=filename) #plot temperature
DataProcesser.output_movie(fns, resolution=100, sigma=density_sigma, filename=f"number_density_{filename}_sigma{density_sigma}.mp4", fps=10, plot_func="plot_gas_number_density",filepath=filename) #plot number density



end_time=datetime.datetime.today()
print('end_time=',datetime.datetime.today())
print('time_cost=',end_time-start_time)
