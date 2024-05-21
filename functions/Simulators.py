import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numba as nb
# from Environment import Environment
from Particles import Particles
# from DataProcesser import DataProcesser

class Simulators:
    """
    Class simulator is base on the simulator we have on class. Which can simulate the evolution of particles.
    """
    def __init__(self, particle: Particles):
        #TODO
        self.particle = particle
        # self.data = DataProcesser()
        # self.Environment = Environment()

    def evolve(self, dt:float, tmax:float, collision=False, resume=False):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """
        #TODO
        self.time = 0
        self.tmax = 10
        self.time_arr = np.linspace(self.time, tmax, int(tmax/dt)+1)   
        # future work: try to read the last output file and resume if something unexpected interrupted the simulation
        # such as: if (previous output exist == true && resume == true), then start from last output
        
        for i in range(len(self.time_arr)):
            self.next_step(dt)
            if collision:
                self.next_step_collision(dt)
           

    def next_step(self, dt:float):
        """
        This function will calculate the next step of the simulation.
        """
        #TODO
        self.time += dt
        self.particle.pos += self.particle.vel * dt


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
    tmax = 10
    
    particles_number = 10
    particles = Particles(particles_number)
    particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,molecular_weight=28.9) 
    
    print(particles.pos)
    print(particles.vel)
    simulation = Simulators(particles)
    simulation.evolve(dt=dt, tmax=tmax, collision=False, resume=False)


