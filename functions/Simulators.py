import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numba as nb
from Environment import Environment
from Particles import Particles
from DataProcesser import DataProcesser

class Simulators:
    """
    Class simulator is base on the simulator we have on class. Which can simulate the evolution of particles.
    """
    def __init__(self, particle: Particles):
        #TODO
        self.particle = Particles
        self.time = 0;
        # future work: try to read the last output file and resume if something unexpected interrupted the simulation
        

    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """
        #TODO

    def next_step(self, dt:float):
        """
        This function will calculate the next step of the simulation.
        """
        #TODO
        self.time += dt

    def next_step_collision(self, dt:float):
        """
        This function will calculate the next step of the simulation with collision.
        """
        #TODO
        pass

if __name__ == "__main__":
    
    nthreads = 2
    nb.set_num_threads(nthreads)
    
    particles_number = 10
    particles = Particles(particles_number)
    particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,molecular_weight=28.9) 
    
    simulation = Simulators(particles)
    simulation.evolve(dt=0.01, tmax=10)
