import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numba as nb
from envtest import Environment
from Particles import Particles
from DataProcesser import DataProcesser

class Simulators:
    """
    Class simulator is base on the simulator we have on class. Which can simulate the evolution of particles.
    """
    def __init__(self, particle: Particles, data: DataProcesser, env: Environment):
        #TODO
        self.particle = particle
        self.data = data
        self.env = env

    def evolve(self, dt=0.01, collision=False):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """
        #TODO
        
        # future work: try to read the last output file and resume if something unexpected interrupted the simulation
        # such as: if (previous output exist == true && resume == true), then start from last output
        
        self.next_step(dt)
        if collision:
            self.next_step_collision(dt)
           

    def next_step(self, dt:float):
        """
        This function will calculate the next step of the simulation.
        """
        #TODO
        self.particle.pos += self.particle.vel * dt
        self.particle.pos, self.particle.vel = self.env.boundary_bounce(self.particle, room_size=self.env.room_size, in_bound=True)


    def next_step_collision(self, dt:float):
        """
        This function will calculate the next step of the simulation with collision.
        """
        #TODO
        pass

if __name__ == "__main__":
    
    nthreads = 2
    nb.set_num_threads(nthreads)

    dt = 0.01
    t_init = 0
    tmax = 10
    
    particles_number = 10
    particles = Particles(particles_number)
    particles.set_particles(pos_type='uniform', vel_type='Boltzmann', \
                            room_size=[0,50,0,50], T=300, molecular_weight=28.9) 
    
    print("init pos:\n", particles.pos)
    print()
    print("init vel:\n" ,particles.vel)
    print()

    particles.vel[:, 0] *= 0.01
    particles.vel[:, 1] *= 0.01

    data = DataProcesser(1, 2) 
    
    envir = Environment(room_size=[0,50,0,50],  \
                        heat_zone_size=[25,25,25,25], open_window=2)
    
    simulation = Simulators(particles, data, envir)
    
    time_arr = np.linspace(t_init, tmax, int((tmax - t_init) / dt) + 1)

    for i in range(len(time_arr)):
        simulation.evolve(dt=dt, collision=False)
        # if i % 10 == 0:
        #     print("time: ", time_arr[i])
        #     print("pos:\n", particles.pos)
        #     print()
       