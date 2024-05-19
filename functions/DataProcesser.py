import numpy as np
import scipy.constants as const
import scipy.stats as stats
import matplotlib.pyplot as plt
import numba as nb
from scipy.ndimage import gaussian_filter
import pickle
import os
import matplotlib.animation as animation
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
    def plot_gas_number_density(particles, resolution=100,sigma=5,fig_save=False):
        """
        This function will plot the gas density of the particles.
        particles: The particles object.
        resolution: How many bins to divide the room into.Example: resolution=100 means the room is divided into 100*100 bins.
        sigma: The standard deviation of the Gaussian filter. Higer value will smooth the data more.
        """
        x = particles.pos[:, 0]
        y = particles.pos[:, 1]
        # Define resolution and room size
        room_size= particles.room_size
        H, _, _ = np.histogram2d(x, y, bins=resolution)
        smoothed_H = gaussian_filter(H, sigma=sigma)
        vmax=particles.nparticles/(resolution**2)*2
        im=plt.imshow(smoothed_H.T, extent=room_size,vmin=0,vmax=vmax,cmap='gray', origin='lower')
        cb = plt.colorbar()
        cb.set_label('Number of Particles in one bin')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D Gas Number Density Distribution at t='+str(particles.time))
        if fig_save:
            if not os.path.exists('Gas_number_density_Distribution'):
                os.mkdir('Gas_number_density_Distribution')
            plt.savefig(f'Gas_number_density_Distribution/gas_number_density_resolution{resolution}_t{particles.time}.png')
            plt.close()
        else:
            plt.show()
        return im

    @staticmethod
    def plot_gas_temperature(particles, resolution=100,vmin=280,vmax=320,sigma=5,fig_save=False):
        """
        This function will plot the gas temperature of the particles distribution in the room.
        particles: The particles object.
        resolution: How many bins to divide the room into.Example: resolution=100 means the room is divided into 100*100 bins.
        vmin: The minimum value of the colorbar. In this case, it is the minimum value of the temperature.
        vmax: The maximum value of the colorbar. In this case, it is the maximum value of the temperature.
        sigma: The standard deviation of the Gaussian filter. Higer value will smooth the data more.
        -------------------------------------------------------------------------------------------
        Noted:
        higher resolution will make the plot away from the real temperature distribution since there will be less number of particles being average in each bin.
        BUT, the whole point is that you need to make sure you have enough number of particles to average in each bin.
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
        particles.temperature_average=round(temp,2)
        print('T_average=',particles.temperature_average,'K')
        # Plot the 2D histogram with mean values
        im=plt.imshow(smoothed_mean_values.T, extent=(xmin, xmax, ymin, ymax), cmap='coolwarm',vmin=vmin,vmax=vmax, origin='lower')
        plt.colorbar(label='Temperature (K)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D Temperature Distribution at t='+str(particles.time))
        if fig_save:
            if not os.path.exists('Tempturature_Distribution'):
                os.mkdir('Tempturature_distribution')
            plt.savefig(f'Tempturature_distribution/temperature_Distribution_resolution{resolution}_t{particles.time}.png')
            plt.close()
        else:   
            plt.show() 
        return im


  
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
        This function will output the data into a binary file.
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
        time="{:04}".format(particles.time)
        path=f'{filepath}/{filename}_t{time}.bin'
        with open(path, 'wb') as file:
            pickle.dump(particles, file)
        return

    @staticmethod
    def data_input(filepath):
        """
        This function will input the data from the file and return the particles object.
        The file should be a binary file.
        """
        path=f'{filepath}'
        if  not os.path.exists(path):
            return FileNotFoundError(f'The file {path} does not exist. Please check your path.')
        with open( path, 'rb') as file:       
            return pickle.load(file)

    @staticmethod
    def data_plot():
        """
        This function will plot the data.
        """
        #TODO
        pass

    @staticmethod
    def output_movie(fns, filename='movie.mp4',fps=30,plot_func='plot_gas_temperature'):
        """
        This function will create a movie of the data.
        """
        fig = plt.figure()
        def init():
            # Do any necessary setup here
            return []
        def update(frame):
            fn=fns[frame]
            print(fn)
            print(frame)
            particles = DataProcesser.data_input(fn)    
            if plot_func=='plot_gas_number_density':
                return DataProcesser.plot_gas_number_density(particles, resolution=100,sigma=5,fig_save=True),
            elif plot_func=='plot_gas_temperature':
                return DataProcesser.plot_gas_temperature(particles, resolution=100,vmin=280,vmax=320,sigma=5,fig_save=True),
        ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init)
        ani.save(filename, writer='ffmpeg', fps=fps)
        return
    
    @staticmethod
    def load_files(header,pattern='[0-9][0-9][0-9][0-9]'):
        import glob
        """
        Load the data from the output file

        :param header: string, the header of the output file
        """
        fns=f'data/{header}_t{pattern}.bin'
        fns = glob.glob(fns)
        fns.sort()
        print(fns)  
        return fns


    if __name__ == '__main__':
        import numpy as np
        from Particles import Particles
        from DataProcesser import DataProcesser
        import time
        import gc
        from numba import set_num_threads
        gc.collect()
        nthreads = 8
        set_num_threads(nthreads)
        particles_number=10000
        particles=Particles(particles_number)
        particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,molecular_weight=28.9)
        start_time = time.time()
        #DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
        #DataProcesser.plot_position_distribution(particles.pos,room_size=particles.room_size, Nsection=1)
        #DataProcesser.plot_gas_number_density(particles, resolution=100,sigma=3,fig_save=True)
        #DataProcesser.plot_gas_temperature(particles, resolution=100,vmin=280,vmax=320,sigma=7,fig_save=True)
        DataProcesser.data_output(particles,'data','particles')
        #particles=DataProcesser.data_input('data/particles_t0000.bin')
        #DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
        End_time = time.time()
        print('Time:',End_time-start_time)

        fns=load_files('particles')
        DataProcesser.output_movie(fns, filename='movie.mp4',fps=1,plot_func='plot_gas_number_density')
        
 
