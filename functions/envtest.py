import numpy as np
import scipy
from functions.Particles import Particles

class Environment:
    """
    Class environment sets how the particles interact with the environment.
    """    
    def __init__(self, room_size = [0,15,0,10], heat_zone_size = [10,15,0,10], open_window = 2):
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
        # define the range of the window
        window_dot =(heat_zone_size[2] + heat_zone_size[3]) / 2
        self.window = np.array([heat_zone_size[0],heat_zone_size[0],\
                                window_dot - open_window/2, window_dot + open_window/2])

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

    def set_window(self, open_window):
        """
        Can modify the length of the window on the wall blocking heat zone and ac room.
        """
        window_dot =(self.heat_zone[2] + self.heat_zone[3]) / 2
        self.window = np.array([self.heat_zone[0],self.heat_zone[0],\
                                window_dot - open_window/2, window_dot + open_window/2])
        return

    def set_heat_zone(self, heat_zone_size):
        """
        Can modify the size of the heat zone.
        heat_zone_size: [xmin,xmax,ymin,ymax]
        """
        self.heat_zone = np.array(heat_zone_size)
        return

    def wall_between_ac_room_and_heat_space(self):
        room = self.room_size
        window = self.window
        up_wall = np.array([window[0], window[1], window[3], room[3]])
        down_wall = np.array([window[0], window[1], room[2], window[2]])
        return up_wall, down_wall


    def boundary_bounce(self,particle, room_size = None, in_bound = True):
        """
        This function will simulate how particles interact with the boundary(wall).
        particle: a Particles object
        room_size: size of the room [xmin,xmax,ymin,ymax] = self.room_size if not specified
        in_bound: True if the particle is inside the boundary, False if the particle is outside the boundary
        """
        pos = particle.pos
        vel = particle.vel
        if room_size is not None:
            self.room_size = room_size
        r_pos = room_size
        # check if the particle is out of the boundary for in_bound senario
        if in_bound:
            mask1 = pos[:, 0] <= r_pos[0]
            mask2 = pos[:, 0] >= r_pos[1]
            mask3 = pos[:, 1] <= r_pos[2]
            mask4 = pos[:, 1] >= r_pos[3]
        else:
        # check if the particle is in the boundary for out_bound senario
            mask1 = pos[:, 0] >= r_pos[0]
            mask2 = pos[:, 0] <= r_pos[1]
            mask3 = pos[:, 1] >= r_pos[2]
            mask4 = pos[:, 1] <= r_pos[3]
        # check for negative x boundary
        pos[mask1, 0] = 2 * r_pos[0] - pos[mask1, 0]
        vel[mask1, 0] = -vel[mask1, 0]
        # check for positive x boundary
        pos[mask2, 0] = 2 * r_pos[1] - pos[mask2, 0]
        vel[mask2, 0] = -vel[mask2, 0]
        # check for negative y boundary
        pos[mask3, 1] = 2 * r_pos[2] - pos[mask3, 1]
        vel[mask3, 1] = -vel[mask3, 1]
        # check for positive y boundary
        pos[mask4, 1] = 2 * r_pos[3] - pos[mask4, 1]
        vel[mask4, 1] = -vel[mask4, 1]
        return pos, vel

    def heat_zone_add_tmperature(self, particle, T = 310):
        #TODO
        pass

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
        pos = particle.pos
        vel = particle.vel
        blow_hole = self.ac_blow_hole
        # check which particles are in the suck hole
        mask = self.is_the_particle_in_the_zone(pos, self.ac_suck_hole)
        # change the velocity of the particles in the suck hole.
        pos[mask, 0] = np.random.uniform(blow_hole[0], blow_hole[1], size=np.sum(mask))
        pos[mask, 1] = np.random.uniform(blow_hole[2], blow_hole[3], size=np.sum(mask))
        vel[mask, 0] = np.sqrt(8 * kB * T / np.pi / m)
        vel[mask, 1] = 0
        
        return pos, vel
        
    @staticmethod
    def is_the_particle_in_the_zone(pos, zone):
        """
        This function will check if the particle is in the zone.
        pos: position of the particle. np.array([x,y])
        zone: zone of the room. [xmin,xmax,ymin,ymax]
        """
        return (pos[:,0] > zone[0] and pos[:,0] < zone[1] and pos[:,1] > zone[2] and pos[:,1] < zone[3])


    


