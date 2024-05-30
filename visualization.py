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

nthreads = 1

nb.set_num_threads(nthreads)
#'''
#set environment
envir = Environment(room_size=[0,5000,0,5000],heat_zone_size=[2400,2500,0,2500])

filepaths='data/suck_blow_heat_n10000_t0_100_0.1'
filename='suck_blow_heat_n10000'

fns=DataProcesser.load_files(filepaths,filename) 
particles=DataProcesser.data_input(f'{filepaths}/{filename}_t00001.bin')   
DataProcesser.plot_velocity_distribution(particles.count_average_T(), particles.mass, particles.vel, save=True) 
print('t00001_average temperature=',particles.count_average_T())
particles=DataProcesser.data_input(f'{filepaths}/{filename}_t00500.bin')   
DataProcesser.plot_velocity_distribution(particles.count_average_T(), particles.mass, particles.vel, save=True)  
print('t00500_average temperature=',particles.count_average_T())
particles=DataProcesser.data_input(f'{filepaths}/{filename}_t01000.bin')   
DataProcesser.plot_velocity_distribution(particles.count_average_T(), particles.mass, particles.vel, save=True)  
print('t01000_average temperature=',particles.count_average_T())



#output movie  


DataProcesser.output_movie(fns, resolution=200, sigma=10, filename="temperature.mp4", fps=10, plot_func="plot_gas_temperature") #plot temperature

DataProcesser.output_particles_movie(fns, envir.room_size, filename=f'{filename}.mp4', fps=30) #plot particles
#DataProcesser.output_movie(fns, resolution=200, sigma=0, filename="number_density_test.mp4", fps=10, plot_func="plot_gas_number_density") #plot number density
end_time=datetime.datetime.today()
print('end_time=',datetime.datetime.today())
print('time_cost=',end_time-start_time)
