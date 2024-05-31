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

nthreads = 8
nb.set_num_threads(nthreads)
#'''
#set environment
envir = Environment(room_size=[0,5000,0,5000],heat_hole_width=5000,heat_hole_buffer=0.05)
filename='suck_blow_1heat_n100000_dT20'
filepaths=f'data/{filename}_t0_100_0p1'



#new simulation
#set particles
particles_number=100000
particles=Particles(particles_number)
particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=envir.room_size,T=310,particles_radius=3,molecular_weight=28.9)
AC_temperature=290

#set simulation
simulation = Simulators(particles, envir)


#set time step and time intial and final
dt = 0.1
t_init = round(particles.step*particles.dt,2)
print('t_init=',t_init)
tmax = 100


#particles.vel[:, 0] *= 0.01
#particles.vel[:, 1] *= 0.01 


#filepaths=f'data/N{particles_number}_t{t_init}_{dt}_{tmax}_Collision_off_HeatZone_[2400,2500,0,2500]/'
#filename='AC_simulation'


time_arr = np.linspace(t_init, tmax, int((tmax - t_init) / dt) + 1)

for i in range(len(time_arr)):


    particles.vel = envir.heat_hole_add_temperature(particles,T=310)
    particles.pos,particles.vel,particles.step,particles.dt = simulation.evolve(dt=dt, collision=False)
    particles.pos,particles.vel=envir.ac_suck_and_blow(particles,T=AC_temperature)

    particles.vel = envir.ac_suck_behavior(particles)
    if i % 1 == 0:  
        #count average temperature of the particles in each step
        #particles.T=Particles.count_average_T(particles)
        #save data in each steps
       # print('particles.vel=',particles.vel)
        DataProcesser.data_output(particles, filepaths, filename)
        time=datetime.datetime.today()-start_time
        time=time/(i+1)*(len(time_arr)-(i+1))
        print(f'\rtime: {"{:.2f}".format(particles.step*particles.dt)},  {round(i/len(time_arr)*100,2)}% is done,  estimated remaining time = {time}',end='')
#==========================Save movie=================================  
#generate the velocity distribution plot of last step to check the velocity distribution
#DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)


#release memory
gc.collect()
#'''
