import numpy as np
import scipy.constants as const
import scipy.stats as stats
import matplotlib.pyplot as plt
class DataProcesser:
    """
    This class will analyse the data from the simulation.
    """
    def __init__(self, n, m):
        #TODO
        pass
    
    @staticmethod
    def plot_velocity_distribution(temperature:float, mass:float, vel:list):
        """
        This function will plot the velocity distribution of the particles.
        Temperature: The temperature of the room.
        Mass: The mass of the particles.
        vel: The velocity vector of the particles.
        """
        #TODO
        #plot the histogram of the data
        speeds = np.linalg.norm(vel, axis=1) #calculate the magnitude of the speeds
        plt.hist(speeds, bins=50, alpha=0.6, color='g', density=True, label='Simulation') #plot the histogram of the data
        
        #create the theoretical distribution
        speeds_range = np.linspace(0, np.max(speeds), 100) #create 100 points from 0 to max speed
        theory_distribution = stats.maxwell.pdf(speeds_range, scale=np.sqrt(const.k*temperature / mass)) #apply the maxwell distribution
        plt.plot(speeds_range, theory_distribution, 'r--', label='Theory')
        plt.title('Speed Distribution')
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_position_distribution(pos:list,room_size:list, Nsection=5):
        """
        This function will plot the position distribution of the particles.
        Position: The position of the particles.
        room_size: The size of the room [xmin,xmax,ymin,ymax].
        Nsection: The number of sections to divide the room.
        """
        xmin=room_size[0]
        xmax=room_size[1]
        ymin=room_size[2]
        ymax=room_size[3]
        bins_x=int((xmax-xmin)/Nsection)
        bins_y=int((ymax-ymin)/Nsection)
        
        

        x_average=np.ones(bins_x)*len(pos[:,0])/bins_x
        y_average=np.ones(bins_x)*len(pos[:,1])/bins_y
        x=np.linspace(xmin,xmax,bins_x)
        y=np.linspace(ymin,ymax,bins_x)
        plt.figure(figsize=(8,12), dpi=80)
        ax1 = plt.subplot(211)
        ax1.plot(x, x_average, 'r--', label='Average x')
        ax1.hist(pos[:,0], bins=bins_x, alpha=0.6, color='g', label='Simulation') #plot the histogram of the x position
        ax1.set_title('x position Distribution')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('Number of Particles')
        ax1.legend()
        ax2 = plt.subplot(212)
        ax2.plot(y, y_average, 'r--', label='Average y')  
        ax2.hist(pos[:,1], bins=bins_y, alpha=0.6, color='b', label='Simulation') #plot the histogram of the y position
        ax2.set_title('y position Distribution')
        ax2.set_xlabel('y (m)')
        ax2.set_ylabel('Number of Particles')
        ax2.legend()

        plt.show()
        
        

 
    @staticmethod
    def data_output():
        """
        This function will output the data to the file.
        """
        #TODO
        pass

    @staticmethod
    def data_input():
        """
        This function will save the data to the file.
        """
        #TODO
        pass

    @staticmethod
    def data_plot():
        """
        This function will plot the data.
        """
        #TODO
        pass

    @staticmethod
    def data_movie():
        """
        This function will create a movie of the data.
        """
        #TODO
        pass