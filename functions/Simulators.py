import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numba as nb
from functions.envtest import Environment
# from Environment import Environment
from functions.Particles import Particles
from functions.DataProcesser import DataProcesser
import os
from scipy.spatial.distance import pdist, squareform
class Simulators:
    """
    Class simulator is base on the simulator we have on class. Which can simulate the evolution of particles.
    """
    def __init__(self, particles: Particles, env: Environment):
        #TODO
        self.particles = particles
        self.env = env

    def evolve(self, dt=0.01, collision=False):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        :param collision: bool, whether the particles collide with each other
        :param auto_heat: bool, whether the particles are heated by the environment
        """
        #TODO
        # check whether particles bounce on the wall
        self.particles.pos, self.particles.vel = self.env.boundary_bounce(self.particles, room_size=self.env.room_size, in_bound=True)
        if collision:
            critical_distance = 2 * self.particles.particles_radius 
            self.particles.pos,self.particles.vel=Simulators.collision(self.particles.pos,self.particles.vel,self.particles.nparticles,critical_distance)
        self.next_step(dt)
        self.particles.step = int(self.particles.step + 1)
        self.particles.dt=dt
        
        return self.particles.pos,self.particles.vel,self.particles.step,self.particles.dt
           
 
    def next_step(self,dt:float):
        """
        This function will calculate the next step of the simulation.
        """
        # get the next position and velocity of the particles
        self.particles.pos += self.particles.vel * dt 



    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def collision(pos, vel, nparticles, critical_distance: float):
        """
        This function will calculate the next step of the simulation with collision.
        """
        # 找到碰撞的小球 (dist < 2 * self.r)
        # dist[i][j] 表示小球 i 和 j 的距离
        for i in range(nparticles):
            for j in range(nparticles):
                if i>j and i!=j:
                    dist = np.linalg.norm(pos[i] - pos[j])
                    if  dist < 2 * critical_distance:
                    # 刚体碰撞速度公式，考虑弹性碰撞以及相同质量做简化
                        pos_i, vel_i = pos[i], vel[i]
                        pos_j, vel_j = pos[j], vel[j]
                        r_ij, v_ij = pos_i - pos_j, vel_i - vel_j
                        r_dot_r = r_ij @ r_ij
                        v_dot_r = v_ij @ r_ij
                        Jn = -v_dot_r * r_ij / r_dot_r
                        vel[i] += Jn
                        vel[j] -= Jn
                else:
                    continue

        return pos, vel
    
if __name__ == "__main__":
    
    nthreads = 2
    nb.set_num_threads(nthreads)
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

    #particles.vel[:, 0] *= 0.1
    #particles.vel[:, 1] *= 0.1 
    
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
            plt.scatter(particles.pos[:, 0], particles.pos[:, 1], )
            plt.title(f"Particles distribution, t = {time_arr[i]}")
            #plt.show()

            
            if  not os.path.exists('functions/figures'):
                os.mkdir('functions/figures')
            plt.savefig(f"functions/figures/Particles_distribution_{time_arr[i]}.png")
    
            DataProcesser.data_output(particles, "data", "test_rotate")