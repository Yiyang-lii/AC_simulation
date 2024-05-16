import numpy as np

from functions.Particles import Particles
from functions.DataProcesser import DataProcesser
from functions.Environment import Environment
from functions.Simulators import Simulators


particles_number=10000
particles=Particles(particles_number)
particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,particle_type='air')
DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
DataProcesser.plot_position_distribution(particles.pos,room_size=particles.room_size, Nsection=1)
