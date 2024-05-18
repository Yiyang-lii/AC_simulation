import numpy as np
import scipy.constants as const
import scipy.stats as stats
class Particles:
    """
    Class particles is the base on the particles we have on class. It contains the basic properties of particles.
    It contains the following properties:
    nparticles: number of particles 
    time: time of the simulation
    pos: position of the particles
    vel: velocity of the particles
    mass: mass of the particles
    T:  average temperature of all particles
    room_size: size of the room.(only support rectangle room)
    pos_type: type of position distribution
    vel_type: type of velocity distribution
    molecular_weight: molecular weight of the particles
    particles_radius: radius of the particles
    """
    def __init__(self,n):
        """
        Create empty lists for every property of the particles.
        """
        self.nparticles = n
        self.time=0
        self.pos=np.zeros((n,2))
        self.vel=np.zeros((n,2))
        return

    
    def pos_distrib(self):
        """
        This function will distribute the particles randomly in certain distribution in the room.
        Some parameters are needed to calculate the distribution:
        n: number of particles
        room_size: size of the room [xmin,xmax,ymin,ymax]
        type: type of distribution (uniform, normal)
        """
        room_size=self.room_size
        n=self.nparticles
        type=self.pos_type
        xmin,xmax,ymin,ymax=room_size
        if type=='uniform':
            self.pos[:,0]= np.random.uniform(xmin,xmax,(n)) 
            self.pos[:,1]= np.random.uniform(ymin,ymax,(n))
        elif type=='normal':
            self.pos[:,0]= np.random.normal((xmin+xmax)/2,(xmax-xmin)/6,(n))
            self.pos[:,1]= np.random.normal((ymin+ymax)/2,(ymax-ymin)/6,(n))
        return



    def vel_distrib(self):
        """
        This function will calculate the distribution velocity (temperature) of the room.
        Some parameters are needed to calculate the distribution:
        T: temperature of the room
        m: mass of the particles
        N: number of particles
        vel_type: type of distribution (Boltzmann)
        """
        T=self.T
        m=self.mass
        N=self.nparticles
        v_type=self.vel_type
        k=const.Boltzmann
        if v_type=='Boltzmann':
            speed = stats.maxwell.rvs(scale=np.sqrt(k*T/ m), size=N) #scale=a in https://mathworld.wolfram.com/MaxwellDistribution.html
            #detailed from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.maxwell.html#ra6d46ce10274-1
            theta = np.random.uniform(0, 2*np.pi, N) #random angle
            self.vel[:,0] = speed*np.cos(theta)
            self.vel[:,1] = speed*np.sin(theta)
        else:
            return ValueError('Distribution type not supported')
        return
    
    def set_particles(self,pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,molecular_weight=28.9,particles_radius='None'):
        """
        This function will set the properties of particles.
        pos_type: type of position distribution (uniform, normal)
        vel_type: type of velocity distribution (Boltzmann)
        room_size: size of the room [xmin,xmax,ymin,ymax]
        T: temperature of the particles(K) ex.room temperature=300
        molecular_weight: molecular weight of the particles(amu) ex.air=28.9
        particles_radius: radius of the particles(0.1nm=1e-10m) ex.air=1.55
        """
        self.mass=const.physical_constants['atomic mass constant'][0]*molecular_weight
        self.particles_radius=particles_radius
        self.T=T
        self.room_size=room_size
        self.pos_type=pos_type
        self.vel_type=vel_type
        self.molecular_weight=molecular_weight
        self.vel_distrib()
        self.pos_distrib()
        return
    
    def count_average_T(self):
        """
        This function will calculate the average temperature of the particles.
        """
        return np.mean(np.linalg.norm(self.vel, axis=1)**2*self.mass/(3*const.Boltzmann))
    
    if __name__ == "__main__":
        import numpy as np
        from Particles import Particles
        from DataProcesser import DataProcesser


        particles_number=100000
        particles=Particles(particles_number)
        particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,molecular_weight=28.9)
        print(particles.count_average_T())
        pass