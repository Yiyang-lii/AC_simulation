import numpy as np
import scipy.constants as const
import scipy.stats as stats
import matplotlib.pyplot as plt
import numba as nb
from scipy.ndimage import gaussian_filter
import pickle
import os
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
    def plot_gas_density(x:list, y:list, room_size:list, resolution=100,sigma=5):
        """
        This function will plot the gas density of the particles.
        x: The x position of the particles.
        y: The y position of the particles.
        room_size: The size of the room [xmin,xmax,ymin,ymax].
        resolution: How many bins to divide the room into.Example: resolution=100 means the room is divided into 100*100 bins.
        sigma: The standard deviation of the Gaussian filter. Higer value will smooth the data more.
        """
        H, _, _ = np.histogram2d(x, y, bins=resolution)
        smoothed_H = gaussian_filter(H, sigma=sigma)
        plt.imshow(smoothed_H.T, extent=room_size, cmap='gray', origin='lower')
        cb = plt.colorbar()
        cb.set_label('Counts in Bin')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('Smoothed 2D Histogram with Specified Resolution')
        plt.show()

    def plot_gas_temperature(particles, resolution=200,vmin=0,vmax=1000,sigma=5):
        """
        This function will plot the gas temperature of the particles distribution in the room.
        particles: The particles object.
        resolution: How many bins to divide the room into.Example: resolution=100 means the room is divided into 100*100 bins.
        vmin: The minimum value of the colorbar. In this case, it is the minimum value of the temperature.
        vmax: The maximum value of the colorbar. In this case, it is the maximum value of the temperature.
        """
        x = particles.pos[:, 0]
        y = particles.pos[:, 1]
        tempturature = np.linalg.norm(particles.vel, axis=1)**2*particles.mass/(3*const.Boltzmann)
        # Define resolution and room size
        xmin, xmax, ymin, ymax = particles.room_size
        # Calculate bin edges
        xbins = np.linspace(xmin, xmax, resolution)
        ybins = np.linspace(ymin, ymax, resolution)
        # Loop through each bin and calculate the mean value of z_values within that bin
        mean_values=DataProcesser.determine_bins_z_value(xbins,ybins,x,y,tempturature)
        # Smooth the data using a Gaussian filter
        smoothed_mean_values = gaussian_filter(mean_values, sigma=sigma)
        #ckeck the mean value of the temperature
        temp=np.mean(tempturature)
        print('T_average=',round(temp,3),'K')
        # Plot the 2D histogram with mean values
        plt.imshow(smoothed_mean_values.T, extent=(xmin, xmax, ymin, ymax), cmap='jet',vmin=vmin,vmax=vmax, origin='lower')
        plt.colorbar(label='Mean Z Values')
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('2D Histogram with Mean Z Values')
        plt.show() 
        return temp

  
    @staticmethod
    @nb.njit(parallel=True)
    def determine_bins_z_value(xbins,ybins,x,y,z):
        '''
        This function will determine the meanvalue of z in each bin and return the mean value as a 2D array .
        '''
        # Calculate bin centers
        xcenters = (xbins[:-1] + xbins[1:]) / 2
        ycenters = (ybins[:-1] + ybins[1:]) / 2
        # Initialize an array to store the mean values
        mean_values = np.zeros((len(xcenters), len(ycenters)))
        i=range(len(xcenters))
        j=range(len(ycenters))
        for i in range(len(xcenters)):
           for j in range(len(ycenters)):
               x_in_bin = (x >= xbins[i]) & (x < xbins[i+1])
               y_in_bin = (y >= ybins[j]) & (y < ybins[j+1])
               points_in_bin = x_in_bin & y_in_bin
               if np.sum(points_in_bin) > 0:
                   mean_values[i, j] = np.mean(z[points_in_bin])
        return mean_values
 
    @staticmethod
    def data_output(particles, filepath,filename):
        """
        This function will output the data to the file.
        """
        if  not os.path.exists(filepath):
            try:
                os.mkdir(filepath)
                print(f'Seems like the folder of {filepath} does not exist. Creating it now.')
            except :
                print(f'Over one folder does not exist along {filepath}. Please check your path.\n \
                      To prevent the data from deleted, the file will be saved in the  folder "data" under current directory.')
                if  not os.path.exists('./data'):
                    os.mkdir('./data')
                    print(f'Creating the folder "data" under current directory.')
                else:
                    filepath='./data'       
        path=filepath+'/'+filename+'.bin'
        with open(path, 'wb') as file:
            pickle.dump(particles, file)
        return

    @staticmethod
    def data_input(object,filepath,filename):
        """
        This function will save the data to the file.
        """
        path=filepath+'/'+filename+'.bin'
        if  not os.path.exists(path):
            return FileNotFoundError(f'The file {path} does not exist. Please check your path.')
        with open( path, 'rb') as file:
            object = pickle.load(file)        
        return object

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
    if __name__ == '__main__':
        import numpy as np
        from Particles import Particles
        from DataProcesser import DataProcesser
        import time
        import gc
        gc.collect()
        particles_number=100000
        particles=Particles(particles_number)
        particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,molecular_weight=28.9)
        #DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
        #DataProcesser.plot_position_distribution(particles.pos,room_size=particles.room_size, Nsection=1)
        #DataProcesser.plot_gas_density(particles.pos[:,0], particles.pos[:,1], particles.room_size, resolution=100,sigma=3)
        DataProcesser.plot_gas_temperature(particles, resolution=200,vmin=0,vmax=1000,sigma=5)
        DataProcesser.data_output(particles,'data','particles')
        particles=DataProcesser.data_input(particles,'data','particles')
 
