import numpy as np
import scipy
from .Particles import Particles

class Environment:
    """
    Class environment sets how the particles interact with the environment.
    """    
    def __init__( self, room_size, acs_hole_area, acb_hole_area, heat_zone):
        """
        Define the properties of the environment.
        room_size: size of the room (only support rectangle room) [xmin,xmax,ymin,ymax]
        acs_hole_pos: position of the air conditioner sucking hole [xmin,xmax,ymin,ymax]
        acb_hole_pos: position of the air conditioner blowing hole [xmin,xmax,ymin,ymax]
        heat_zone: size of the area that increase temperature of particles [xmin,xmax,ymin,ymax]
        """
        self.room_size     = np.array(room_size)
        self.acs_hole_area = np.array(acs_hole_area)
        self.acb_hole_area = np.array(acb_hole_area)
        self.acs_hole_dot  = np.array([(acs_hole_area[0]+acs_hole_area[1])/2,\
                                      (acs_hole_area[2]+acs_hole_area[3])/2])
        self.heat_zone     = np.array(heat_zone)
        


    def wall_bounce(self,particle):
        """
        This function will simulate how particles interact with the boundary(wall) .
        particle: a Particles object
        """
        num = particle.nparticles
        pos = particle.pos
        vel = particle.vel
        r_pos = self.room_size
        # start the loop to check every particle in the particles
        for n in range(num):
            # check if the particle is out of the x boundary
            if pos[n,0] < r_pos[0]:
                vel[n,0] = -vel[n,0]
                pos[n,0] = 2 * r_pos[0] - pos[n,0] 
            elif pos[n,0] > r_pos[1]:
                vel[n,0] = -vel[n,0]
                pos[n,0] = 2 * r_pos[1] - pos[n,0]
            # check if the particle is out of the y boundary
            if pos[n,1] < r_pos[2]:
                vel[n,1] = -vel[n,1]
                pos[n,1] = 2 * r_pos[2] - pos[n,1]
            elif pos[n,1] > r_pos[3]:
                vel[n,1] = -vel[n,1]
                pos[n,1] = 2 * r_pos[3] - pos[n,1]
        return pos, vel

    def air_con_suck(self,particle):
        """
        This function will simulate the air conditioner sucking air
        """
        dot     = self.acs_hole_dot
        area    = self.acs_hole_area
        n       = particle.nparticles
        pos     = particle.pos
        vel     = particle.vel
        r       = pos - dot
        dic_rot = np.cross(pos,r)
        # start the loop to check every particle in the particles
        vel = Particles.rotate_particles()
            



    def air_con_blow(self):
        """
        This function will simulate the air conditioner blowing air
        """
        #TODO
        pass

    def heat_storage(self,T):
        """
        This function will simulate the hot spot in the room.
        """
        #TODO
        pass


