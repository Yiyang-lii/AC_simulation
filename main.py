import numpy as np
import numba as nb
from functions.Particles import Particles
from functions.DataProcesser import DataProcesser
from functions.Environment import Environment
from functions.Simulators import Simulators
import gc
import datetime
print('start_time=',datetime.datetime.today())
nthreads = 1

nb.set_num_threads(nthreads)
particles_number=10000
particles=Particles(particles_number)
particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=envir.room_size,T=310,particles_radius=1,molecular_weight=28.9)
   

#set simulation
simulation = Simulators(particles, envir)


#set time step and time intial and final
dt = 0.1
t_init = 0
tmax = 5


#particles.vel[:, 0] *= 0.01
#particles.vel[:, 1] *= 0.01 
filepaths='data/N100000_t0_0.1_100_Collision_off_HeatZone_on'
filename='AC_simulation'

#filepaths=f'data/N{particles_number}_t{t_init}_{dt}_{tmax}_Collision_off_HeatZone_[2400,2500,0,2500]/'
#filename='AC_simulation'


time_arr = np.linspace(t_init, tmax, int((tmax - t_init) / dt) + 1)

for i in range(len(time_arr)):
    particles.dt=dt
    particles.step = i
    simulation.evolve(dt=dt, collision=False)
    particles.pos,particles.vel=envir.ac_suck_and_blow(particles,T=290)
    #particles.vel=envir.heat_zone_add_temperature(particles,T=310)

    if i % 1 == 0:  
        print("time: ", "{:.2f}".format(particles.step*particles.dt))
        #count average temperature of the particles in each step
        particles.T=Particles.count_average_T(particles)
        #save data in each step
        DataProcesser.data_output(particles, filepaths, filename)
#generate the velocity distribution plot of last step to check the velocity distribution
DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
DataProcesser.plot_position_distribution(particles.pos,room_size=particles.room_size, Nsection=1)
