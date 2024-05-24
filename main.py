import numpy as np
import numba as nb
from functions.Particles import Particles
from functions.DataProcesser import DataProcesser
from functions.Environment import Environment
from functions.Simulators import Simulators
import gc
nthreads = 8
nb.set_num_threads(nthreads)
filepaths='data/test_rotate'
filename='AC_simulation'
#set environment
envir = Environment(room_size=[0,5000,0,5000],heat_zone_size=[25,25,25,25], open_window=2)


#set particles
particles_number=100000
particles=Particles(particles_number)
particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=envir.room_size,T=300,particles_radius=1,molecular_weight=28.9)
   

#set simulation
simulation = Simulators(particles, envir)


#set time step and time intial and final
dt = 0.1
t_init = 0
tmax = 100


#particles.vel[:, 0] *= 0.01
#particles.vel[:, 1] *= 0.01 





time_arr = np.linspace(t_init, tmax, int((tmax - t_init) / dt) + 1)

for i in range(len(time_arr)):
    particles.dt=dt
    particles.step = i
    simulation.evolve(dt=dt, collision=True)
    envir.ac_suck_and_blow(particles)
    envir.heat_zone_add_temperature(particles,T=310)

    if i % 1 == 0:  
        print("time: ", time_arr[i])  
        #count average temperature of the particles in each step
        particles.T=Particles.count_average_T(particles)
        #save data in each step
        DataProcesser.data_output(particles, filepaths, filename)
#generate the velocity distribution plot of last step to check the velocity distribution
DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)


#release memory
gc.collect()

#==========================Save movie=================================
#load data
fns=DataProcesser.load_files(filepaths,filename)      
#output movie  
DataProcesser.output_movie(fns, resolution=200, sigma=5, filename="temperature.mp4", fps=10, plot_func="plot_gas_temperature") #plot temperature
DataProcesser.output_movie(fns, resolution=200, sigma=5, filename="number_density.mp4", fps=10, plot_func="plot_gas_number_density") #plot number density