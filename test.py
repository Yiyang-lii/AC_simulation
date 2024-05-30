<<<<<<< Updated upstream
import random
=======
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
filepaths='data/test'
filename='AC_simulation'
particles=DataProcesser.data_input(f'{filepaths}/{filename}_t000124.bin')   
'''
#set particles
particles_number=10000
particles=Particles(particles_number)
particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=envir.room_size,T=310,particles_radius=10,molecular_weight=28.9)
'''
#set simulation
simulation = Simulators(particles, envir)
>>>>>>> Stashed changes

# 產生包含數字1、2、3的列表
numbers = ['C', 'L', 'G']

# 洗牌列表
random.shuffle(numbers)

# 顯示隨機順序的數字
for index, number in enumerate(numbers):
    print(f"{index + 1}: {number}")