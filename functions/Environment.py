import numpy as np
import scipy
from .Particles import Particles

class Environment:
    """
    Class environment sets how the particles interact with the environment.
    """    
    def __init__(self, room_size = [0,10,0,10], heat_zone_size = [10,15,0,10]):
        """
        Define the properties of the environment.
        room_size: size of the room (only support rectangle room) [xmin,xmax,ymin,ymax]
        heat_zone: size of the area that increase temperature of particles [xmin,xmax,ymin,ymax]
        """
        self.room_size     = np.array(room_size)
        self.heat_zone     = np.array(heat_zone_size)        
        # define default ac suck hole
        half_hole_length        = (room_size[3] - room_size[2])/4
        buff_length             = half_hole_length*0.1
        suck_hole_mid_dot       = (3*room_size[3] + room_size[2])/4
        self.suck_zone_radius   = half_hole_length * 4
        self.ac_suck_hole_dot   = np.array([room_size[0], suck_hole_mid_dot])
        self.ac_suck_hole       = np.array([room_size[0], room_size[0]+ buff_length,\
                                suck_hole_mid_dot - half_hole_length, suck_hole_mid_dot + half_hole_length])
        # define default ac blow hole
        blow_hole_mid_dot = (3*room_size[2] + room_size[3])/4
        self.ac_blow_hole_dot   = np.array([room_size[0], blow_hole_mid_dot])
        self.ac_blow_hole       = np.array([room_size[0], room_size[0] + buff_length,\
                                blow_hole_mid_dot - half_hole_length, blow_hole_mid_dot + half_hole_length])

    def set_room_size(self, room_size):
        """
        Can modify the size of the room.
        room_size: [xmin,xmax,ymin,ymax]
        """
        self.room_size = np.array(room_size)
        return

    def set_ac_suck_hole(self, ac_suck_hole):
        """
        Can modify the position of the air conditioner suck hole.
        ac_suck_hole: [xmin,xmax,ymin,ymax]
        """
        self.ac_suck_hole       = np.array(ac_suck_hole)
        self.ac_suck_hole_dot   = np.array([(ac_suck_hole[0] + ac_suck_hole[1])/2, \
                                            (ac_suck_hole[2] + ac_suck_hole[3])/2])
        return
        
    def set_suck_zone_radius(self, radius: float):
        """
        Can modify the radius of the air conditioner suck zone.
        radius: radius of the suck zone
        """
        self.suck_zone_radius = radius
        return

    def set_ac_blow_hole(self, ac_blow_hole):
        """
        Can modify the position of the air conditioner blow hole.
        ac_blow_hole: [xmin,xmax,ymin,ymax]
        """
        self.ac_blow_hole       = np.array(ac_blow_hole)
        self.ac_blow_hole_dot   = np.array([(ac_blow_hole[0] + ac_blow_hole[1])/2, \
                                            (ac_blow_hole[2] + ac_blow_hole[3])/2])
        return

    def set_heat_zone(self, heat_zone_size):
        """
        Can modify the size of the heat zone.
        heat_zone_size: [xmin,xmax,ymin,ymax]
        """
        self.heat_zone = np.array(heat_zone_size)
        return

    def wall_bounce(self,particle):
        #TODO: add a window and a wall between ac room and heat space
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

    def ac_suck_behavior(self,particle):
        """
        This function will simulate the air conditioner sucking air. Which will turn the velocity of the 
        particles more towards the air conditioner suck hole.
        particle: a Particles object
        """
        source = self.ac_suck_hole_dot
        radius = self.suck_zone_radius
        pos = particle.pos
        vel = particle.vel
        return Particles.rotate_particles(pos,vel,radius,source)

    def ac_suck_and_blow(self, particle, T = 298):
        """
        This function will simulate the air conditioner sucking air and blowing air.
        particle: a Particles object
        T : temperature of the air ac blows out
        """
        kB = 1.38064852e-23
        m  = particle.mass
        num = particle.nparticles
        pos = particle.pos
        vel = particle.vel
        blow_hole = self.ac_blow_hole
        # check how much particles should be sucked in
        for n in range(num):
            if self.is_the_particle_in_the_zone(pos, self.ac_suck_hole, n):
                pos[n,0] = np.random.uniform(blow_hole[0], blow_hole[1])
                pos[n,1] = np.random.uniform(blow_hole[2], blow_hole[3])
                vel[n,0] = np.sqrt(8*kB*T/np.pi/m)
                vel[n,1] = 0
        return pos, vel

    def heat_zone_add_tmperature(self, particle, T = 310):
        #TODO
        pass
        
        
    @staticmethod
    def is_the_particle_in_the_zone(pos, zone, n):
        """
        This function will check if the particle is in the zone.
        pos: position of the particle. np.array([x,y])
        zone: zone of the room. [xmin,xmax,ymin,ymax]
        """
        return (pos[n,0] > zone[0] and pos[n,0] < zone[1] and pos[n,1] > zone[2] and pos[n,1] < zone[3])


    


