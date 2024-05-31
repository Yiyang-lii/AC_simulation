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
    def plot_velocity_distribution(temperature:float, mass:float, vel:list, save=False,filepath='velocity_distribution',filename='velocity_distribution'):
        """
        This function will plot the velocity distribution of the particles.
        Temperature: The temperature of the room.
        Mass: The mass of the particles.
        vel: The velocity vector of the particles.
        """
        #TODO
        #plot the histogram of the data
        mask = ~np.isnan(vel).any(axis=1)
        vel =  vel[mask]
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
        if save==True:
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            plt.savefig(f'{filepath}/{filename}.png')
            print('save') 
        plt.close()
        return

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
        plt.close()
    @staticmethod
    def plot_gas_number_density(particles, resolution=100,sigma=5,fig_save=False,filepath='Number_Density_Distribution',filename='Number_Density_Distribution'):
        """
        This function will plot the gas density of the particles. It will return the image object of the plot.
        particles: The particles object.
        resolution: How many bins to divide the room into.Example: resolution=100 means the room is divided into 100*100 bins.
        sigma: The standard deviation of the Gaussian filter. Higer value will smooth the data more.
        """
        mask = ~np.isnan(particles.pos).any(axis=1)
        particles.pos =  particles.pos[mask]
        x = particles.pos[:, 0]
        y = particles.pos[:, 1]
        # Define resolution and room size
        room_size= particles.room_size
        H, _, _ = np.histogram2d(x, y ,bins=resolution)
        smoothed_H = gaussian_filter(H, sigma=sigma)
        vmax=particles.nparticles/(resolution**2)*2
        im=plt.imshow(smoothed_H.T, extent=room_size,vmin=0,vmax=vmax,cmap='gray', origin='lower')
        cb = plt.colorbar()
        cb.set_label('Number of Particles in one bin')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D Gas Number Density Distribution at t='+str("{:.2f}".format(particles.step*particles.dt)))
        if fig_save=='video':
            return 
        elif fig_save==True:
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            plt.savefig(f'{filepath}/{filename}.png')
        elif fig_save==False:
            plt.show()
        plt.close()
        return im

    @staticmethod
    def plot_gas_temperature(particles, resolution=100,vmin=280,vmax=320,sigma=1,fig_save=False,filepath='Temperature_Distribution',filename='Temperature_Distribution'):
        """
        This function will plot the gas temperature of the particles distribution in the room. It will return the image object of the plot.
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
        plt.title('2D Temperature Distribution at t='+str("{:.2f}".format(particles.step*particles.dt)))
        if fig_save=='video':
            return 
        elif fig_save==True:
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            plt.savefig(f'{filepath}/{filename}.png')
        elif fig_save==False:
            plt.show()
        plt.close()
        return im

    @staticmethod
    def plot_particles(particles):
        """
        This function will plot the particles.
        particles: The particles object.
        """
        plt.figure()
        plt.scatter(particles.pos[:, 0], particles.pos[:, 1], s=1)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Particles Distribution')
        plt.show()
        plt.close()
        pass
  
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
        for i in nb.prange(len(xcenters)):
           for j in nb.prange(len(ycenters)):
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
        particles: The particles object.
        filepath: The path of the file. ex. 'data'
        filename: The name of the file. ex. 'particles' 
        """
        if  not os.path.exists(filepath):
            os.makedirs(filepath)
        step="{:05}".format(particles.step)
        path=f'{filepath}/{filename}_t{step}.bin'
        with open(path, 'wb') as file:
            pickle.dump(particles, file)
        return

    @staticmethod
    def data_input(filepath):
        """
        This function will input the data from the file and return the particles object.
        The file should be a binary file.
        filepath: The path of the file. ex. 'data/particles_t00000.bin'
        """
        path=f'{filepath}'
        if  not os.path.exists(path):
            return FileNotFoundError(f'The file {path} does not exist. Please check your path.')
        with open( path, 'rb') as file:       
            return pickle.load(file)
        
    def output_particles_movie(fns, filename='movie.mp4', fps=30,filepath='video'):
        plt.style.use('dark_background')
        fig, ax = plt.subplots()
        fig.set_linewidth(5)
        fig.set_size_inches(10, 10, forward=True)
        fig.set_dpi(72)
        line, = ax.plot([], [], '.', color='w', markersize=5)

        def init():
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            Particles=DataProcesser.data_input(fns[0])
            ax.set_xlim(Particles.room_size[0],Particles.room_size[1])
            ax.set_ylim(Particles.room_size[2],Particles.room_size[3])
            ax.set_aspect('equal')
            ax.set_xlabel('X [code unit]', fontsize=18)
            ax.set_ylabel('Y [code unit]', fontsize=18)
            return line, 

        def update(frame):
            fn = fns[frame]
            Particles=DataProcesser.data_input(fn)
            line.set_data(Particles.pos[:,0], Particles.pos[:, 1])
            plt.title("Frame =" + str(frame), size=18)
            print('frame=',frame)
            return line,
        ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init, blit=True)
        filepath=f'{filepath}/{filename}'
        ani.save(filepath, writer='ffmpeg', fps=fps)
        return
    @staticmethod
    def output_movie(fns, resolution=100, sigma=5, filename='movie.mp4', fps=30, plot_func='plot_gas_temperature',filepath='video'):
        """
        This function will create a movie of the data.
        fns: list, the list of output files
        filename: string, the name of the output movie
        fps: int, the frame per second of the movie
        plot_func: string, the name of the plot function
        """
        fig = plt.figure()
        def init():
            # Do any necessary setup here

            return []
        def update(frame):
            plt.clf()
            fn = fns[frame]
            particles = DataProcesser.data_input(fn)  
            print(f'\r frame = {frame}, step={particles.step}, filepath = {fn}   ', end='')
            if plot_func == 'plot_gas_temperature':
                DataProcesser.plot_gas_temperature(particles, resolution=resolution,sigma=sigma,vmin=270,vmax=330,fig_save='video')
            elif plot_func == 'plot_gas_number_density':
                DataProcesser.plot_gas_number_density(particles, resolution=resolution,sigma=sigma,fig_save='video')
            return

        ani = animation.FuncAnimation(fig, update, frames=len(fns), init_func=init)
        filepath=f'{filepath}/{filename}'
        ani.save(filepath, writer='ffmpeg', fps=fps)
        return

    
    @staticmethod
    def load_files(filepath,header,pattern='[0-9][0-9][0-9][0-9][0-9]'):
        import glob
        """
        Load the data from the output file
        filepath: string, the path of the output file ex. 'data'
        header: string, the header of the output file

        """
        fns=f'{filepath}/{header}_t{pattern}.bin'
        fns = glob.glob(fns)
        fns.sort()
        #print('fns=',fns)
        return fns

    @staticmethod
    def plot_multi_zones(zone_size = [np.array([0,1,0,1])], color = ['r'], zone_only=True):
        """
        This finction will plot the zones in the room.
        zone_size: list with numpy list as elements, the size of the zone ex. [np.[xmin,xmax,ymin,ymax],...]
        n_zone: int, the number of zones
        color: list with string as elements, the color of the zone ex. ['r','b',...]
        zone_only: bool, True to create a new figure, False to plot on the current figure.

        more information about the zone_size:
        1. If the zone is a rectangle, the zone_size should be a numpy array with 4 elements [xmin,xmax,ymin,ymax]
        2. If the zone is a circle, the zone_size should be a numpy array with 3 elements [radius,center_position,angle_range]
        """
        # if we only want to plot the zones
        if zone_only:
            plt.figure()
        # run the main loop for ploting zones
        for n, zone in enumerate(zone_size):
            if len(zone) == 4:  
                x = [zone[0],zone[1],zone[1],zone[0]]
                y = [zone[2],zone[2],zone[3],zone[3]]
                plt.fill(x,y,color[n])
            elif len(zone) == 3:
                theta = np.linspace(zone[2][0], zone[2][1], 100)
                x = zone[1][0] + zone[0] * np.cos(theta)
                y = zone[1][1] + zone[0] * np.sin(theta)
                plt.fill(x, y, color[n])
            else:
                print(f'{zone} is a invalid zone size. Please check your zone size.')
                continue
        # close the plot if we only want to plot the zones
        if zone_only:
            plt.show()
        return

    @staticmethod
    def plot_points_on_zones(points, color='black'):
        """
        This function will plot the points on the zones.
        points: list with numpy list as elements, the position of the points ex. [np.array([x,y]),...]
        color: string, the color of the points
        To use this function, you need to open a plt.figure() first. Since it won't create a new figure.
        """
        for x, y in points:
            plt.scatter(x, y, color=color)
        return
    def plot_temperature_versus_time(fns,filepath, filename, save=True):
        """
        This function will plot the temperature versus time.
        particles: The particles object.
        filepath: The path of the file. ex. 'data'
        filename: The name of the file. ex. 'particles' 
        """
        temperature = []
        t=[]
        for fn in fns:
            particles = DataProcesser.data_input(fn)
            T=particles.count_average_T()
            if np.isnan(T):
                continue
            else:
                temperature.append(T)
                t.append(particles.step*particles.dt)
        plt.plot(t,temperature)
        plt.xlabel('Time(s)')
        plt.ylabel('Temperature(K)')
        plt.title('Temperature versus Time')
        if save:
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            plt.savefig(f'{filepath}/{filename}.png')
        return

        


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
    #particles_number=10000
    #particles=Particles(particles_number)
    #particles.set_particles(pos_type='uniform',vel_type='Boltzmann',room_size=[0,50,0,50],T=300,molecular_weight=28.9)
    start_time = time.time()
    #DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
    #DataProcesser.plot_position_distribution(particles.pos,room_size=particles.room_size, Nsection=1)
    #DataProcesser.plot_gas_number_density(particles, resolution=100,sigma=3,fig_save=False)
    #DataProcesser.plot_gas_temperature(particles, resolution=100,vmin=280,vmax=320,sigma=7,fig_save=True)
    # DataProcesser.data_output(particles,'data','particles')
    #particles=DataProcesser.data_input('../data/test_rotate_t0000.bin')
    #DataProcesser.plot_velocity_distribution(particles.T, particles.mass, particles.vel)
    End_time = time.time()
    print('Time:',End_time-start_time)
    fns=DataProcesser.load_files('test_rotate')
    DataProcesser.output_movie(fns, resolution=100, sigma=5, filename="test_rotate.mp4", fps=5, plot_func="plot_gas_number_density")
    
    
 
