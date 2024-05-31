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

#set environment

room_length = 2500
particles_number=100000
AC_temperature = 290
room_temperature = 310
heat_hole_width_scale = 1
dt = 0.1
tmax = 100

heat_hole_width_scale_tag=str(heat_hole_width_scale).replace('.','p')
dt_tag=str(dt).replace('.','p')
print('heat_hole_width_scale_tag=',heat_hole_width_scale_tag)
print('dt_tag=',dt_tag)
filename=f'{heat_hole_width_scale_tag}heat_n{particles_number}_room{room_length}_dT{room_temperature-AC_temperature}'
filepaths=f'data/{filename}_t_0_{tmax}_{dt_tag}'

#set environment
envir = Environment(room_size=[0,room_length,0,room_length],heat_hole_width=room_length*heat_hole_width_scale)

particles=Particles(particles_number)
particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=envir.room_size,T=room_temperature,particles_radius=3,molecular_weight=28.9)

simulation = Simulators(particles, envir)

#set time intial 
t_init = round(particles.step*particles.dt,2)
print('t_init=',t_init)



#particles.vel[:, 0] *= 0.01
#particles.vel[:, 1] *= 0.01 

time_arr = np.linspace(t_init, tmax, int((tmax - t_init) / dt) + 1)

for i in range(len(time_arr)):
    particles.vel = envir.heat_hole_add_temperature(particles,T=room_temperature, zone=False)
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


#release memory
gc.collect()
#'''
