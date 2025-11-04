import h5py as h5
import numpy as np
import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import re
from sklearn.decomposition import FastICA, PCA    
import os
from scipy.stats import spearmanr
from cmcrameri import cm
import multiprocessing as multi
import warnings
import copy


def get_roma():
    return cm.roma.reversed()

def get_c_norm(data):
    # Determine the maximum absolute value
    max_abs_value = np.nanmax(np.abs(data))

    print(max_abs_value)
    
    return mcolors.TwoSlopeNorm(vmin=-max_abs_value, vcenter=0, vmax=max_abs_value)

def update_file_path(old_path):
    # Split the path into directory and file name
    directory, filename = os.path.split(old_path)
    
    # Modify the file name
    new_filename = filename.replace('cum.h5', 'cum_filt.h5')
    
    # Combine the directory and new file name to get the new path
    new_path = os.path.join(directory, new_filename)
    
    return new_path

class h5_data:
    def __init__(self,h5_filename):
        with h5.File(h5_filename,'r') as cumh5:
            vel = cumh5['vel'] # this is the average velocity??
            cum = cumh5['cum'] # This is the cumulitive displacement ??
            self.n_im, self.length, self.width = cum.shape
            
            imdates = cumh5['imdates'][()].astype(str).tolist()
            imdates_dt = np.array(([dt.datetime.strptime(imd, '%Y%m%d') for imd in imdates])) ##datetime
            imdates_ordinal = np.array(([dt.datetime.strptime(imd, '%Y%m%d').toordinal() for imd in imdates])) ##73????
            imdates_ordinal = imdates_ordinal + mdates.date2num(np.datetime64('0000-12-31'))
        
            
            refarea = cumh5['refarea'][()] # FROMS LiCSBAS 
            if type(refarea) is bytes:
                refarea = refarea.decode('utf-8')
            self.refx1, self.refx2, self.refy1, self.refy2 = [int(s) for s in re.split('[:/]', refarea)]
            refx1, refx2, refy1, refy2 = self.refx1, self.refx2, self.refy1, self.refy2

            try:
                self.mask = cumh5["mask"][()]
                self.mask[self.mask==0] = np.nan
                self.ts_ref = np.nanmean(cum[:, refy1:refy2, refx1:refx2]*self.mask[refy1:refy2, refx1:refx2], axis=(1, 2))
                ts_ref = self.ts_ref[:, np.newaxis, np.newaxis]
                self.cum = cum - ts_ref
                self.cum_mask = self.cum*self.mask
                print("mask applied from cum.h5 file")
            except:
                try:
                    print(update_file_path(h5_filename))
                    with h5.File(update_file_path(h5_filename),'r') as cum_filth5:
                        self.mask = cum_filth5["mask"][()]
                        self.mask[self.mask==0] = np.nan
                        self.ts_ref = np.nanmean(cum[:, refy1:refy2, refx1:refx2]*self.mask[refy1:refy2, refx1:refx2], axis=(1, 2))
                        ts_ref = self.ts_ref[:, np.newaxis, np.newaxis]
                        self.cum = cum - ts_ref
                        self.cum_mask = self.cum*self.mask
                        print("mask applied from cum_filt.h5 file")
                except:
                    self.ts_ref = np.nanmean(cum[:, refy1:refy2, refx1:refx2], axis=(1, 2))
                    ts_ref = self.ts_ref[:, np.newaxis, np.newaxis]
                    self.cum = cum - ts_ref
                    self.cum_mask = self.cum
                    print("mask not found and therefore not applied")
                
            self.x = imdates_ordinal-imdates_ordinal[0]
            self.x_dates = [datetime.strptime(date, "%Y%m%d") for date in imdates]

            self.lat1 = float(cumh5['corner_lat'][()])###########ADDED
            self.lon1 = float(cumh5['corner_lon'][()])
            self.dlat = float(cumh5['post_lat'][()])
            self.dlon = float(cumh5['post_lon'][()])###########ADDED

        self.h5_filename = h5_filename
        directory, filename = os.path.split(h5_filename)
        
        try:
            # Combine the directory and new file name to get the new path
            hgt_path = os.path.join(directory, 'hgt')

            self.hgt = np.fromfile(hgt_path, dtype=np.float32).reshape(self.length, self.width)
            self.hgt[np.isnan(self.hgt)] = 0
        except:
            hgt_path = os.path.join(directory, 'results', 'hgt')

            self.hgt = np.fromfile(hgt_path, dtype=np.float32).reshape(self.length, self.width)
            self.hgt[np.isnan(self.hgt)] = 0
            
    def get_ts(self,y,x,masked=False):
        if masked:
            return self.cum_mask[:,y,x]
        else:
            return self.cum[:,y,x]

    def get_multiple_ts(self,y_range,x_range,masked=False):
        if masked:
            return self.cum_mask[:,y_range[0]:y_range[1],x_range[0]:x_range[1]]
        else:
            return self.cum[:,y_range[0]:y_range[1],x_range[0]:x_range[1]]

    def get_cum(self,masked=False):
        if masked:
            return self.cum_mask
        else:
            return self.cum
import pandas as pd



class ICA_area:
    def __init__(self,h5_object,volcano_name,y_range=False,x_range=False):
        self.x_range=x_range
        self.y_range=y_range
        self.volcano_name=volcano_name
        

        self.x=h5_object.x
        self.x_dates=h5_object.x_dates
        
        if y_range == False or x_range == False:
            self.cum=h5_object.get_cum()
            self.cum_masked=h5_object.get_cum(True)
            self.hgt = h5_object.hgt
        else:
            self.cum=h5_object.get_multiple_ts(self.y_range,self.x_range)
            self.cum_masked=h5_object.get_multiple_ts(self.y_range,self.x_range,True)
            self.hgt=h5_object.hgt[y_range[0]:y_range[1],x_range[0]:x_range[1]]

        if self.x_range == False or self.y_range == False:
            self.x_range = (0, self.cum.shape[2])
            self.y_range = (0, self.cum.shape[1])

        self.n_images, self.length, self.width = self.cum.shape
        self.flatten()

        self.lat1 = h5_object.lat1
        self.dlat = h5_object.dlat
        self.lon1 = h5_object.lon1
        self.dlon = h5_object.dlon

        lats = np.arange(self.y_range[0], self.y_range[-1])
        lons = np.arange(self.x_range[0], self.x_range[-1])
        
        self.lats, temp = xy2bl(lats,lats, self.lat1, self.dlat, self.lon1, self.dlon)
        temp,self.lons = xy2bl(lons,lons, self.lat1, self.dlat, self.lon1, self.dlon)

        self.dd={}

        self.h5_filename = h5_object.h5_filename
        self.independent_component_labels = False
        self.principal_component_labels = False

    def flatten(self):
        self.cum_flattened=[]
        self.rows=[]
        self.columns=[]
        self.hgt_flattened=[]

        self.cum_flattened_masked=[]
        self.rows_masked=[]
        self.columns_masked=[]
        self.hgt_flattened_masked=[]
        
        for y_c in range(self.y_range[1]-self.y_range[0]):
            for x_c in range(self.x_range[1]-self.x_range[0]):
                ts_temp=self.cum[:,y_c,x_c]
                
                if not(type(ts_temp)==float or np.isnan(ts_temp).any()):
                    self.cum_flattened.append(ts_temp)
                    self.columns.append(x_c)
                    self.rows.append(y_c)
                    self.hgt_flattened.append(self.hgt[y_c][x_c])

                ts_temp_masked=self.cum_masked[:,y_c,x_c]
                
                if not(type(ts_temp_masked)==float or np.isnan(ts_temp_masked).any()):
                    self.cum_flattened_masked.append(ts_temp_masked)
                    self.columns_masked.append(x_c)
                    self.rows_masked.append(y_c)
                    self.hgt_flattened_masked.append(self.hgt[y_c][x_c])

        self.cum_flattened=np.array(self.cum_flattened)
        self.cum_flattened_masked=np.array(self.cum_flattened_masked)
    
    def cast_to_2d(self,data,masked=False):
        new_data = np.empty((self.y_range[-1]-self.y_range[0], self.x_range[-1]-self.x_range[0]))
        new_data[:]=np.nan
        if masked:
            for i,d in enumerate(data):
                new_data[self.rows_masked[i]][self.columns_masked[i]]=d
            return new_data
        else:
            for i,d in enumerate(data):
                new_data[self.rows[i]][self.columns[i]]=d
            return new_data
        

    def get_flattened_index(self,x,y,masked=False):
        if masked:
            return np.where(np.logical_and(np.array(self.rows_masked)==y,np.array(self.columns_masked)==x))[0][0]
        else:
            return np.where(np.logical_and(np.array(self.rows)==y,np.array(self.columns)==x))[0][0]

    
    def plot_cumulitive_displacement(self,masked=False):
        if masked:
            to_plot=self.cast_to_2d(self.cum_flattened_masked[:,-1]-self.cum_flattened_masked[:,0],True)
        else:
            to_plot=self.cast_to_2d(self.cum_flattened[:,-1]-self.cum_flattened[:,0])
        plt.imshow(to_plot,cmap=get_roma())
        plt.colorbar()
        plt.show()

    def plot_cumulitive_displacement_ax(self,ax,masked=False):
        if masked:
            im=ax.imshow(self.cast_to_2d(self.cum_flattened_masked[:,-1]-self.cum_flattened_masked[:,0],masked), cmap=get_roma())
            rect=plt.Rectangle((self.columns_masked[self.index_of_max]-0.5,self.rows_masked[self.index_of_max]-0.5),1,1,edgecolor='lime',facecolor='none')
        else:
            im=ax.imshow(self.cast_to_2d(self.cum_flattened[:,-1]-self.cum_flattened[:,0],masked), cmap=get_roma())
            rect=plt.Rectangle((self.columns[self.index_of_max]-0.5,self.rows[self.index_of_max]-0.5),1,1,edgecolor='lime',facecolor='none')
        ax.add_patch(rect)
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
            
        ax.set_title("Cumulitive Displacement")
            
        #ax.set_xticks([])
        #ax.set_yticks([])

    #def get_weight_component(self,component_number):
    #    scaled_component = np.tile(self.independent_components[component_number], (len(self.weights[:,component_number]), 1))
    #    w = np.tile(self.weights[:,component_number], (len(self.independent_components[component_number]), 1)).T
    #    #print(scaled_component)
    #     #print(w)
    #    return scaled_component * w
    
    def get_full_component(self, component_number, masked=False):
        if masked:
            weights = self.cast_to_2d(self.weights[:,component_number],True)
        else:
            weights = self.cast_to_2d(self.weights[:,component_number])
        component = self.independent_components[component_number][:, np.newaxis, np.newaxis] * weights
        return component

    def perform_ICA(self,n_components,masked=False,clip_range=False,max_iter=2000, tol=0.0001):

        self.n_components=n_components
        self.ica_clip_range=clip_range
        self.ica_masked=masked
        self.ica = FastICA(n_components=self.n_components,max_iter=max_iter,tol=tol)#,whiten=False)
        #self.ica = ica(n_components=self.n_components)
        if not(self.ica_clip_range):
            if self.ica_masked:
                #ica_fit_transform(self.cum_flattened_masked.T,self.ica,n_components_to_keep)  # Transpose for temporal ICA
                self.ica.fit_transform(self.cum_flattened_masked.T)
                independent_components = self.ica.transform(self.cum_flattened_masked.T).T  # Transpose back
            else:
                #ica_fit_transform(self.cum_flattened.T,self.ica,n_components_to_keep) 
                self.ica.fit_transform(self.cum_flattened.T)
                independent_components = self.ica.transform(self.cum_flattened.T).T  # Transpose back

        weights = self.ica.mixing_

        for i in range(self.n_components):
            independent_components[i] = independent_components[i] - independent_components[i][0]

            scale = independent_components[i][-1]
            independent_components[i] = independent_components[i] / scale
            weights[:, i] = weights[:, i] * scale
        
        self.weights = weights
        self.independent_components = independent_components
    
    def plot_ICA(self,representative_index="max",cbar_locked = False, date_list = False, wrapped=False):


        if representative_index in ("max","Max","MAX"):
            if self.ica_masked:
                self.index_of_max=np.argmax(self.cum_flattened_masked[:,-1])
            else:
                self.index_of_max=np.argmax(self.cum_flattened[:,-1])
        elif representative_index in ("min", "Min", "MIN"):
            if self.ica_masked:
                self.index_of_max=np.argmin(self.cum_flattened_masked[:,-1])
            else:
                self.index_of_max=np.argmin(self.cum_flattened[:,-1])
        else:
            self.index_of_max = self.get_flattened_index(representative_index[0],representative_index[1],self.ica_masked)

        print(self.index_of_max)

        fig, axes = plt.subplots(self.n_components+1, 3, figsize=(15+12, 3 * (self.n_components+1)),gridspec_kw={'width_ratios': [3, 2, 2]})
            
        for i, (ax, comp) in enumerate(zip(axes, self.independent_components)):
            ax[0].plot(self.x_dates, comp, label=f"{i+1}th component")
            ax[0].set_title(f"{i+1}th ICA Component of {self.volcano_name} Displacement")
            ax[0].set_xlabel("Date")
            ax[0].set_ylabel("Value (arb)")
            if date_list:
                for dates in date_list:
                    ax[0].vlines(dates.date_list, min(comp),max(comp) ,color=dates.date_color, linestyle=dates.date_style, label=dates.date_meaning)
            ax[0].legend()

            weight = self.cast_to_2d(self.weights[:,i],self.ica_masked)
            
            if cbar_locked and not wrapped:
                if cbar_locked == True:
                    cbar_locked = (3,200)
                mean = np.mean(self.weights[:,i])
                std = np.std(self.weights[:,i])
                im=ax[1].imshow(weight, cmap=get_roma(),vmin = max(cbar_locked[1], mean - cbar_locked[0]*std ), vmax = min(cbar_locked[1], mean + cbar_locked[0]*std))
            else:
                plot_colourbar_latlon(ax[1],weight, self.lats, self.lons, self.y_range, self.x_range, cmap = get_roma(), centre_0 = True, label = "LOS Weight (mm)", wrapped=wrapped)
                #im=ax[1].imshow(weight, cmap=get_roma())

            index_of_max = np.nanargmax(weight)
            index_of_max = np.unravel_index(index_of_max, weight.shape)
            index_of_min = np.nanargmin(weight)
            index_of_min = np.unravel_index(index_of_min, weight.shape)

            #print(index_of_max)
            #print(index_of_min)
            
            rect=plt.Rectangle((index_of_max[1]-0.5,index_of_max[0]-0.5),1,1,edgecolor='red',facecolor='none')
            ax[1].add_patch(rect)
            rect=plt.Rectangle((index_of_min[1]-0.5,index_of_min[0]-0.5),1,1,edgecolor='blue',facecolor='none')
            ax[1].add_patch(rect)
            
            #plt.colorbar(im, ax=ax[1], shrink=0.8, pad=0.02)
        
            ax[1].set_title("Mixing Matrix")
        
            #ax[1].set_xticks([])
            #ax[1].set_yticks([])

            if self.ica_masked:
                plot_regression_and_spearman(ax[2], self.hgt_flattened_masked, self.weights[:, i])
            else:
                plot_regression_and_spearman(ax[2], self.hgt_flattened, self.weights[:, i])
            ax[2].set_xlabel("Elevation (m)")
            ax[2].set_ylabel("Weight (arbs)")
            
        if self.ica_masked:
            axes[-1][0].plot(self.x_dates, self.cum_flattened_masked[self.index_of_max], label="Example TS")
            plot_regression_and_spearman(axes[-1][2], self.hgt_flattened_masked, self.cum_flattened_masked[:,-1])
            to_plot=self.cast_to_2d(self.cum_flattened_masked[:,-1]-self.cum_flattened_masked[:,0],True)
        else:
            axes[-1][0].plot(self.x_dates, self.cum_flattened[self.index_of_max], label="Example TS")
            plot_regression_and_spearman(axes[-1][2], self.hgt_flattened, self.cum_flattened[:,-1])
            to_plot=self.cast_to_2d(self.cum_flattened[:,-1]-self.cum_flattened[:,0])

        if wrapped:
            to_plot = wrap_disp(to_plot)

        for i in range(self.n_components):
            axes[-1][0].plot(self.x_dates, self.independent_components[i]*self.weights[self.index_of_max,i], label=f"{i+1}th Component", alpha=0.4)
                
        axes[-1][0].set_title("Whole Time Series")
        axes[-1][0].set_xlabel("Date")
        axes[-1][0].set_ylabel("Displacement")
        axes[-1][0].legend()

        plot_colourbar_latlon(axes[-1][1],to_plot, self.lats, self.lons, self.y_range, self.x_range, cmap = get_roma(), centre_0 = True, label = "Cumulative Displacement (mm)")
        #self.plot_cumulitive_displacement_ax(axes[-1][1],self.ica_masked)

        im=axes[-1][2].imshow(self.hgt,cmap='terrain')
        axes[-1][2].set_title("Height")

        
        plt.tight_layout()
        plt.show()

    def plot_PCA(self,n_components = 20, PCA_masked = False, date_list = False, representative_index = "max"):
        if representative_index in ("max","Max","MAX"):
            if PCA_masked:
                self.index_of_max=np.argmax(self.cum_flattened_masked[:,-1])
            else:
                self.index_of_max=np.argmax(self.cum_flattened[:,-1])
        elif representative_index in ("min", "Min", "MIN"):
            if PCA_masked:
                self.index_of_max=np.argmin(self.cum_flattened_masked[:,-1])
            else:
                self.index_of_max=np.argmin(self.cum_flattened[:,-1])
        else:
            self.index_of_max = self.get_flattened_index(representative_index[0],representative_index[1],masked)
        
        pca = PCA(n_components=n_components)
        
        if PCA_masked:
            principle_components = pca.fit_transform(self.cum_flattened_masked.T).T
        else:
            principle_components = pca.fit_transform(self.cum_flattened.T).T

        pca_weights = pca.components_.T

        fig, axes = plt.subplots(n_components+1, 3, figsize=(15+6, 4 * (n_components+1)),gridspec_kw={'width_ratios': [3, 2, 2]})
            
        for i, (ax, comp) in enumerate(zip(axes, principle_components)):
            ax[0].plot(self.x_dates, comp, label=f"{i+1}th component")
            ax[0].set_title(f"{i+1}th PCA Component of {self.volcano_name} Displacement")
            ax[0].set_xlabel("Date")
            ax[0].set_ylabel("Value (arb)")
            if date_list:
                for dates in date_list:
                    ax[0].vlines(dates.date_list, min(comp),max(comp), color=dates.date_color, linestyle=dates.date_style, label=dates.date_meaning)
            ax[0].legend()
            im=ax[1].imshow(self.cast_to_2d(pca_weights[:,i],PCA_masked), cmap=get_roma())
            plt.colorbar(im, ax=ax[1], shrink=0.8, pad=0.02)
        
            ax[1].set_title("Mixing Matrix")

            if PCA_masked:
                plot_regression_and_spearman(ax[2], self.hgt_flattened_masked, pca_weights[:, i])
            else:
                plot_regression_and_spearman(ax[2], self.hgt_flattened, pca_weights[:, i])
            ax[2].set_xlabel("Elevation (m)")
            ax[2].set_ylabel("Weight (arbs)")
                    
        if PCA_masked:
            index_of_max=np.argmax(self.cum_flattened_masked[:,-1]-self.cum_flattened_masked[:,0])
            axes[-1][0].plot(self.x_dates, self.cum_flattened_masked[index_of_max], label="Example TS")
        else:
            index_of_max=np.argmax(self.cum_flattened[:,-1]-self.cum_flattened[:,0])
            axes[-1][0].plot(self.x_dates, self.cum_flattened[index_of_max], label="Example TS")

        for i in range(n_components):
            axes[-1][0].plot(self.x_dates, principle_components[i]*pca_weights[index_of_max,i], label=f"{i+1}th Component", alpha=0.4)
                
        axes[-1][0].set_title("Whole Time Series")
        axes[-1][0].set_xlabel("Date")
        axes[-1][0].set_ylabel("Displacement")
        axes[-1][0].legend()

        self.plot_cumulitive_displacement_ax(axes[-1][1],PCA_masked)
        
        plt.tight_layout()
        plt.show()
            
    def remove_time(self,index):
        self.cum=np.delete(self.cum,index,0)
        self.cum_masked=np.delete(self.cum_masked,index,0)
        self.x=np.delete(self.x,index)
        self.x_dates=np.delete(self.x_dates,index)
        self.flatten()

    def get_component(self, component_number, pixel_x = False, pixel_y = False, coordinate_of_wholeframe = False, lat_lon = False):
        if pixel_x == False and pixel_y == False:
            return self.independent_components[component_number]
        else:
            if lat_lon:
                pixel_x, pixel_y = bl2xy(pixel_x, pixel_y, self.lat1, self.dlat, self.lon1, self.dlon)
                coordinate_of_wholeframe = True
            if coordinate_of_wholeframe:
                pixel_x = pixel_x - self.x_range[0]
                pixel_y = pixel_y - self.y_range[0]
            weight_to_return = self.cast_to_2d(self.weights[:,component_number],self.ica_masked)
            return self.independent_components[component_number]*weight_to_return[pixel_y,pixel_x]

    def get_cum_ts(self,pixel_x, pixel_y, coordinate_of_wholeframe = False):
        if coordinate_of_wholeframe:
            pixel_x = pixel_x - self.x_range[0]
            pixel_y = pixel_y - self.y_range[0]
            
        if self.ica_masked:
            return self.cum_masked[:,pixel_y,pixel_x]
        else:
            return self.cum[:,pixel_y,pixel_x]
        
    def save_components_for_GBIS(self, output_dir=False, independent_component_labels=False, principal_component_labels=False):
        if independent_component_labels: # set ICA labels if new labels provided
            self.set_component_labels(independent_component_labels, ICA=True)

        if not (principal_component_labels == False and self.principal_component_labels): # set PCA labels unless existing labels and no new labels provided
            self.set_component_labels(principal_component_labels, ICA=False)

        if not output_dir:
            output_dir = os.path.dirname(self.h5_filename)
            output_dir = os.path.join(output_dir, "ICA_components" ,self.volcano_name)
        ica_outpt_file = os.path.join(output_dir, f"ICA_{self.n_components}_components.h5")
        pca_output_file = os.path.join(output_dir, f"PCA_{self.n_components}_components.h5")

        if not os.path.exists(output_dir):
            print("Creating output directory: ", output_dir)
            os.makedirs(output_dir)

        if self.ica_masked:
            residual = self.cum_masked.copy()
        else:
            residual = self.cum.copy()

        print("Saving ICA components to ", ica_outpt_file)
        dt = h5.string_dtype(encoding='utf-8')

        with h5.File(ica_outpt_file, 'w') as ica_h5:

            independent_component_labels=self.independent_component_labels
            independent_component_labels.append('residual')

            ica_h5.create_dataset('component_names', data=np.array(independent_component_labels, dtype=dt))

            ica_h5.create_dataset('corner_lat', data=self.lats[0])
            ica_h5.create_dataset('corner_lon', data=self.lons[0])
            ica_h5.create_dataset('post_lat', data=self.dlat)
            ica_h5.create_dataset('post_lon', data=self.dlon)

            for i in range(self.n_components):
                print("Saving ICA component ", i+1)
                component_to_save = self.get_full_component(i, self.ica_masked)
                residual = residual - component_to_save
                
                #ica_h5.create_dataset(self.independent_component_labels[i], data=component_to_save, compression="gzip")
                ica_h5.create_dataset(self.independent_component_labels[i], data=component_to_save[-1], compression="gzip")

            #ica_h5.create_dataset('residual', data=residual, compression="gzip")
            ica_h5.create_dataset('residual', data=residual[-1], compression="gzip")

        if False: # PCA saving disabled for now

            pca = PCA(n_components=self.n_components)
            
            if PCA_masked:
                principle_components = pca.fit_transform(self.cum_flattened_masked.T).T
            else:
                principle_components = pca.fit_transform(self.cum_flattened.T).T

            pca_weights = pca.components_.T

            print("Saving PCA components to ", pca_output_file)

            with h5.File(pca_output_file, 'w') as pca_h5:

                principal_component_labels=self.principal_component_labels
                principal_component_labels.append('residual')

                pca_h5.create_dataset('component_names', data=np.array(principal_component_labels, dtype='S'))

                pca_h5.create_dataset('corner_lat', data=self.lats[0])
                pca_h5.create_dataset('corner_lon', data=self.lons[0])
                pca_h5.create_dataset('post_lat', data=self.dlat)
                pca_h5.create_dataset('post_lon', data=self.dlon)

                if self.ica_masked:
                    residual = self.cum_masked.copy()
                else:
                    residual = self.cum.copy()

                for i in range(self.n_components):
                    print("Saving PCA component ", i+1)
                    weight = self.cast_to_2d(pca_weights[:,i],self.ica_masked)
                    component_to_save = principle_components[i][:, np.newaxis, np.newaxis] * weight
                    residual = residual - component_to_save
                    
                    pca_h5.create_dataset(self.principal_component_labels[i], data=component_to_save, compression="gzip")

                pca_h5.create_dataset('residual', data=residual, compression="gzip")


    def set_component_labels(self, component_labels, ICA=True):
        if component_labels:
            if len(component_labels) != self.n_components:
                raise ValueError("Length of independent_component_labels must match number of components.")
            used_labels={}
            for i, label in enumerate(component_labels):
                if label in component_labels[:i] or label in component_labels[i+1:]:
                    if label in used_labels:
                        used_labels[label].append(i)
                    else:
                        used_labels[label]=[i]
            
            for label, indices in used_labels.items():
                for count, index in enumerate(indices):
                    component_labels[index] = f"{label}_{count+1}"

        else:
            component_labels = [f"IC_{i+1}" for i in range(self.n_components)]

        if ICA:
            self.independent_component_labels = component_labels
        else:
            self.principal_component_labels = component_labels

    def deramp_wrapper(self,i):
        """
        THIS IS TAKEN FROM LiCSBAS tools_lib.fit2dh WITH MINOR MODIFICATIONS
        Wrapper function for deramping cumulative interferograms.
        """
        cum_org = self.cum[i, :, :]

        if np.mod(i, 10) == 0:
            print("  {0:3}/{1:3}th image...".format(i, len(self.x_dates)), flush=True)

        fit, model = fit2dh(cum_org, self.deg_ramp, self.hgt_flag,
                                    self.hgt_min, self.hgt_max) ## fit is not masked
        _cum = cum_org-fit

        return _cum, model

    def deramp_cum(self,deg_ramp=1, hgt=[], hgt_min=-10000, hgt_max=10000, n_para=1):
        self.deg_ramp = deg_ramp
        self.hgt_flag = hgt
        self.hgt_min = hgt_min
        self.hgt_max = hgt_max

        cum_deramped = np.zeros((self.n_images, self.length, self.width))

        if n_para == 1:
            models = np.zeros(self.n_images, dtype=object)
            for i in range(self.n_images):
                cum_deramped[i, :, :], models[i] = self.deramp_wrapper(i)
        else:
            print('with {} parallel processing...'.format(n_para), flush=True)
            ### Parallel processing
            try:
                p = multi.get_context('fork').Pool(n_para)#Unix-based systems
            except:
                p = multi.get_context('spawn').Pool(n_para)
            _result = np.array(p.map(self.deramp_wrapper, range(self.n_images)), dtype=object)
            p.close()
            del args

            models = _result[:, 1]
            for i in range(self.n_images):
                cum_deramped[i, :, :] = _result[i, 0]
            del _result

        self.cum = cum_deramped
        mask = np.isnan(self.cum_masked[0, :, :]) # assuming the mask is the same for all time steps - quick fix
        self.cum_masked = self.cum * (~mask)

        self.flatten()


class important_dates:
    def __init__(self,date_list, date_meaning, date_color, date_style = "dashed"):
        self.date_style = date_style
        self.date_list = date_list
        self.date_meaning = date_meaning
        self.date_color = date_color
        
def plot_regression_and_spearman(ax, x, y):
    x = np.array(x)
    y = np.array(y)
    # Plot the data points
    ax.plot(x, y, ".")

    # Calculate and plot the linear regression line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, "-", color="red")

    # Calculate Spearman's rank correlation
    spearman_corr, _ = spearmanr(x, y)
    
    # Display Spearman's rank correlation on the plot
    ax.text(0.05, 0.95, f"Spearman's ρ: {spearman_corr:.2f}", transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

#### The Next three function are taken from LiCSBAS

def xy2bl(x, y, lat1, dlat, lon1, dlon):###########ADDED
    """
    xy index starts from 0, end with width/length-1
    """
    lat = lat1+dlat*y
    lon = lon1+dlon*x

    return lat, lon

def bl2xy(lon, lat, lat1, postlat, lon1, postlon):
    """
    lat1 is north edge and postlat is negative value.
    lat lon values are in grid registration
    x/y index start from 0, end with width-1
    """
    x = int(np.round((lon - lon1)/postlon))
    y = int(np.round((lat - lat1)/postlat))

    return [x, y]

def fit2dh(A, deg, hgt, hgt_min, hgt_max, gpu=False):
    """
    Estimate best fit 2d ramp and topography-correlated component simultaneously.

    Inputs:
        A   : Input ndarray (can include nan)
        deg : degree of polynomial for fitting ramp
          - 1  -> a+bx+cy (ramp, default)
          - bl -> a+bx+cy+dxy (biliner)
          - 2  -> a+bx+cy+dxy+ex**2_fy**2 (2d polynomial)
          - []  -> a (must be used with hgt)
        hgt : Input hgt to estimate coefficient of topo-corr component
              If blank, don*t estimate topo-corr component.
        hgt_min : Minimum hgt to take into account in hgt-linear
        hgt_max : Maximum hgt to take into account in hgt-linear
        gpu     : GPU flag

    Returns:
        Afit : Best fit solution with the same demention as A
        m    : Set of parameters of best fit plain (a,b,c...)

    Note: GPU option seems slow and may return error when the size is large.
          Not recommended.

    """
    if gpu:
        import cupy as xp
        A = xp.asarray(A)
        hgt = xp.asarray(hgt)
        hgt_min = xp.asarray(hgt_min)
        hgt_max = xp.asarray(hgt_max)
    else:
        xp = np

    ### Make design matrix G
    length, width = A.shape

    if not deg:
        G = xp.ones((length*width))
    else:
        Xgrid, Ygrid = xp.meshgrid(xp.arange(width), xp.arange(length))
        Xgrid1 = Xgrid.ravel()
        Ygrid1 = Ygrid.ravel()

        if str(deg) == "1":
            G = xp.stack((Xgrid1, Ygrid1)).T
        elif str(deg) == "bl":
            G = xp.stack((Xgrid1, Ygrid1, Xgrid1*Ygrid1)).T
        elif str(deg) == "2":
            G = xp.stack((Xgrid1, Ygrid1, Xgrid1*Ygrid1,
                          Xgrid1**2, Ygrid1**2)).T
        else:
            print('\nERROR: Not proper deg ({}) is used\n'.format(deg), file=sys.stderr)
            return False
        del Xgrid, Ygrid, Xgrid1, Ygrid1

        G = xp.hstack([xp.ones((length*width, 1)), G])

    if len(hgt) > 0:
        _hgt = hgt.copy()  ## Not to overwrite hgt in main
        _hgt[xp.isnan(_hgt)] = 0
        _hgt[_hgt<hgt_min] = 0
        _hgt[_hgt>hgt_max] = 0
        G2 = xp.vstack((G.T, hgt.ravel())).T ## for Afit
        G = xp.vstack((G.T, _hgt.ravel())).T
        del _hgt
    else:
        G2 = G

    G = G.astype(xp.int32)

    ### Invert
    mask = xp.isnan(A.ravel())
    m = xp.linalg.lstsq(G[~mask, :], A.ravel()[~mask], rcond=None)[0]

    Afit = ((xp.matmul(G2, m)).reshape((length, width))).astype(xp.float32)

    if gpu:
        Afit = xp.asnumpy(Afit)
        m = xp.asnumpy(m)
        del A, hgt, hgt_min, hgt_max, length, width, G, G2, mask

    return Afit, m

def wrap_disp(disp, wavelength = 56):
    unw_phase = -4*np.pi*disp/wavelength # -2/wavelenght * 2 pi
    wrapped_phase = np.mod(unw_phase + np.pi, 2*np.pi) - np.pi
    return wrapped_phase

def plot_colourbar_latlon(ax,final_displacement, lats, lons, x_range, y_range, display_pixels = False, cmap='inferno', label='LOS Cumulitive Displacement (mm)', centre_0=False, wrapped=False):
    if wrapped:
        final_displacement = wrap_disp(final_displacement)
    extent = [lons[0], lons[-1], lats[-1], lats[0]]  # min lon, max lon, min lat, max lat
    if centre_0:
        im = ax.imshow(final_displacement, cmap=cmap, aspect='equal',interpolation='nearest', extent=extent, norm=get_c_norm(final_displacement))
    else:
        im = ax.imshow(final_displacement, cmap=cmap, aspect='equal',interpolation='nearest', extent=extent)
    

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Secondary axes for pixel numbers
    def lon_to_pixel(lon):
        return np.interp(lon, [lons[0], lons[-1]], [x_range[0], x_range[-1]-1])

    def pixel_to_lon(pix):
        return np.interp(pix, [x_range[0], x_range[-1]-1], [lons[0], lons[-1]])

    def lat_to_pixel(lat):
        # Now pixel 0 is at the top, so invert mapping
        return np.interp(lat, [lats[0], lats[-1]], [y_range[0], y_range[-1]-1])

    def pixel_to_lat(pix):
        return np.interp(pix, [y_range[0], y_range[-1]-1], [lats[0], lats[-1]])
    
    

    if display_pixels:
        plt.colorbar(im, ax=ax, label=label, pad=0.13)
        secax_x = ax.secondary_xaxis('top', functions=(lon_to_pixel, pixel_to_lon))
        secax_x.set_xlabel('Pixel X')

        secax_y = ax.secondary_yaxis('right', functions=(lat_to_pixel, pixel_to_lat))
        secax_y.set_ylabel('Pixel Y')
    else:
        plt.colorbar(im, ax=ax, label=label)
