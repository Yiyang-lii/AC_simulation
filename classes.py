class Particles:
    """
    Class particles is the base on the particles we have on class. It contains the basic properties of particles.
    """
    def __init__(self, n):
        """
        Create empty lists for every property of the particles.
        """
        #TODO
        pass

    def pos_distrib(self,n,type='normal'):
        """
        This function will distribute the distribution particles in the room.
        """
        #TODO
        pass

    def vel_distrib(self,T,n,type='Boltzmann'):
        """
        This function will calculate the distribution velocity (temperature) of the room.
        """
        #TODO
        pass

    def set_particles(self, pos, vel, acc):
        """
        This function will set the properties of particles.
        """
        #TODO
        pass

    

class Simulators:
    """
    Class simulator is base on the simulator we have on class. Which can simulate the evolution of particles.
    """
    def __init__(self, n, m):
        #TODO
        pass

    def evolve(self, dt:float, tmax:float):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """
        #TODO
        pass

    def dounce_wall(self):
        """
        This function will simulate the particles bouncing off the wall.
        """
        #TODO
        pass

class Environment:
    """
    Class environment sets how the particles interact with the environment.
    """    
    def __init__(self, n, m):
        #TODO
        pass

    def boundary(self):
        """
        This function will simulate how particles interact with the boundary.
        """
        #TODO
        pass

    def air_con_suck(self):
        """
        This function will simulate the air conditioner sucking air
        """
        #TODO
        pass

    def air_con_blow(self):
        """
        This function will simulate the air conditioner blowing air
        """
        #TODO
        pass

class Analyser:
    """
    This class will analyse the data from the simulation.
    """
    def __init__(self, n, m):
        #TODO
        pass

