import numpy as np
import numba as nb
from functions.Particles import Particles
from functions.DataProcesser import DataProcesser
from functions.Environment import Environment
from functions.Simulators import Simulators

nthreads = 8
nb.set_num_threads(nthreads)
particles_number=10000
particles=Particles(particles_number)
particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,particles_radius=1.55,molecular_weight=28.9)
   

dt = 0.01
t_init = 0
tmax = 10
particles_number = 100000
particles = Particles(particles_number)
particles.set_particles(pos_type='uniform', vel_type='Boltzmann', \
                        room_size=[0,50,0,50], T=300, molecular_weight=28.9) 

print("init pos:\n", particles.pos)
print()
print("init vel:\n", particles.vel)
print()

#particles.vel[:, 0] *= 0.01
#particles.vel[:, 1] *= 0.01 

envir = Environment(room_size=[0,50,0,50],  \
                    heat_zone_size=[25,25,25,25], open_window=2)

simulation = Simulators(particles, envir)

time_arr = np.linspace(t_init, tmax, int((tmax - t_init) / dt) + 1)

for i in range(len(time_arr)):
    particles.step = i
    simulation.evolve(dt=dt, collision=False)
    if i % 10 == 0:
        print("time: ", time_arr[i])
        print("vel:\n", particles.vel)
        print()

    if i % 50 == 0:    
        #plt.scatter(particles.pos[:, 0], particles.pos[:, 1], )
        #plt.title(f"Particles distribution, t = {time_arr[i]}")
        #plt.show()

        
        #if  not os.path.exists('functions/figures'):
        #    os.mkdir('functions/figures')
        #plt.savefig(f"functions/figures/Particles_distribution_{time_arr[i]}.png")
        particles.T=Particles.count_average_T(particles)
        DataProcesser.data_output(particles, "data", "test_rotate")
particles=DataProcesser.data_input('data/test_rotate_t1000.bin')

DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
fns=DataProcesser.load_files('test_rotate')        
DataProcesser.output_movie(fns, resolution=100, sigma=5, filename="test_rotate.mp4", fps=2, plot_func="plot_gas_temperature")