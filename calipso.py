#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on 2021-09-07

Copyright (c) 2021 Erik Johansson

Author(s):
Erik Johansson <erik.johansson@lmd.ipsl.fr>
'''

import numpy as np
import h5py  # @UnresolvedImport
import time
from datetime import datetime
import pdb



# READING DATA FROM CALIOP:
scip_these_larger_variables_until_needed = {
    # if any of these are needed just rempve them from the dictionary!
    "Spacecraft_Position": True,  # 3D-variable
    # 2-D variable with second dimension larger than 40:
    "Attenuated_Backscatter_Statistics_1064": True,
    "Attenuated_Backscatter_Statistics_532": True,
    "Attenuated_Total_Color_Ratio_Statistics": True,
    "Volume_Depolarization_Ratio_Statistics": True,
    "Particulate_Depolarization_Ratio_Statistics": True,
    "Cirrus_Shape_Parameter": True,
    "Cirrus_Shape_Parameter_Invalid_Points": True,
    "Cirrus_Shape_Parameter_Uncertainty": True
}


atrain_match_names = {
    # Use version 3 "nsidc_surface_type" as name for sea/ice info
    "Snow_Ice_Surface_Type": "nsidc_surface_type",
    # Add "_tai" to the profile_time name to not forget it is tai time
    "Profile_Time": "profile_time_tai"}



class DataObject(object):
    """
    Class to handle data objects with several arrays.

    """

    def __getattr__(self, name):
        try:
            return self.all_arrays[name]
        except KeyError:
            raise AttributeError("%s instance has no attribute '%s'" % (
                self.__class__.__name__, name))

    def __setattr__(self, name, value):
        if name == 'all_arrays':
            object.__setattr__(self, name, value)
        else:
            self.all_arrays[name] = value

    def __add__(self, other):
        """Adding two objects together"""
        # Check if we have an empty object
        # modis objects does not have longitude attribute
        is_empty_self = True
        is_empty_other = True
        for key in self.all_arrays.keys():
            if self.all_arrays[key] is not None and len(self.all_arrays[key]) > 0:
                is_empty_self = False
        for key in other.all_arrays.keys():
            if other.all_arrays[key] is not None and len(other.all_arrays[key]) > 0:
                is_empty_other = False
        if is_empty_self:
            # print("First object is None!, returning second object")
            return other
        if is_empty_other:
            # print("Second object is None!, returning first object")
            return self
        for key in self.all_arrays:
            try:
                if self.all_arrays[key].ndim != self.all_arrays[key].ndim:
                    raise ValueError("Can't concatenate arrays " +
                                     "of different dimensions!")
            except AttributeError:
                # print "Don't concatenate member " + key + "... " + str(e)
                self.all_arrays[key] = other.all_arrays[key]
                continue
            try:
                if self.all_arrays[key].ndim == 1:
                    self.all_arrays[key] = np.concatenate(
                        [self.all_arrays[key],
                         other.all_arrays[key]])
                elif key in ['segment_nwp_geoheight',
                             'segment_nwp_moist',
                             'segment_nwp_pressure',
                             'segment_nwp_temp']:
                    self.all_arrays[key] = np.concatenate(
                        [self.all_arrays[key],
                         other.all_arrays[key]], 0)
                elif self.all_arrays[key].ndim == 2:
                    self.all_arrays[key] = np.concatenate(
                        [self.all_arrays[key],
                         other.all_arrays[key]], 0)
            except ValueError:
                # print "Don't concatenate member " + key + "... " + str(e)
                self.all_arrays[key] = other.all_arrays[key]
        return self



class CalipsoObject(DataObject):
    def __init__(self):
        DataObject.__init__(self)
        self.all_arrays = {
            # Normal name = calipso.name.lower()

            # Imager matching needed for all truths:
            'longitude': None,
            'latitude': None,
            'imager_linnum': None,
            'imager_pixnum': None,
            'elevation': None,  # DEM_elevation => elevation in (m)"
            'cloud_fraction': None,
            'validation_height': None,
            'sec_1970': None,
            'minimum_laser_energy_532': None,
            'layer_top_altitude': None,
            'layer_top_temperature': None,
            'layer_top_pressure': None,
            'midlayer_temperature': None,
            'layer_base_altitude': None,
            'layer_base_pressure': None,
            'number_layers_found': None,
            'igbp_surface_type': None,
            'nsidc_surface_type': None,  # V4 renamed from 'snow_ice_surface_type'
            'snow_ice_surface_type': None,
            # 'nsidc_surface_type_texture': None,
            'profile_time_tai': None,  # renamed from "Profile_Time"
            'feature_classification_flags': None,
            'day_night_flag': None,
            'feature_optical_depth_532': None,
            'tropopause_height': None,
            'profile_id': None,

            # If a combination of 5 and 1km data are used for RESOLUTION=1
            # "column_optical_depth_tropospheric_aerosols_1064_5km": None,
            # "column_optical_depth_tropospheric_aerosols_1064": None,
            "column_optical_depth_tropospheric_aerosols_532_5km": None,
            "column_optical_depth_tropospheric_aerosols_532": None,
            "column_optical_depth_aerosols_532_5km": None,
            "column_optical_depth_aerosols_532": None,
            # "column_optical_depth_tropospheric_aerosols_uncertainty_1064_5km": None,
            # "column_optical_depth_tropospheric_aerosols_uncertainty_532_5km": None,
            "column_optical_depth_cloud_532_5km": None,
            # "column_optical_depth_cloud_uncertainty_532_5km": None,
            "feature_optical_depth_532_5km": None,
            "layer_top_altitude_5km": None,
            "layer_top_pressure_5km": None,
            "number_layers_found_5km": None,
            # Variables derived for 5km data
            # Also included if a combination of 5 and 1km data are used for RESOLUTION=1
            'detection_height_5km': None,
            'total_optical_depth_5km': None,
            "feature_optical_depth_532_top_layer_5km": None,
            'cfc_single_shots_1km_from_5km_file': None,
            "average_cloud_top_pressure_single_shots": None,
            "average_cloud_top_pressure_single_shots_5km": None,
            "average_cloud_top_single_shots": None,
            "average_cloud_top_single_shots_5km": None,
            "average_cloud_base_single_shots": None,
            "average_cloud_base_single_shots_5km": None,
            "single_shot_data": None,
            # Variables derived from 5km file to 1kmresolution_
            'cfc_single_shots_1km_from_5km_file': None,

            # From cloudsat:
            'cal_modis_cflag': None,
            'cloudsat_index': None,
        }



def rearrange_calipso_the_single_shot_info(retv, singleshotdata):

    # Extract number of cloudy single shots (max 15)
    # plus average cloud base and top
    # in 5 km FOV
    name = "ssNumber_Layers_Found"
    data = singleshotdata[name]
    data = np.array(data)
    data_reshaped_15 = data.reshape(-1, 15)
    single_shot_cloud_cleared_array = np.sum(data_reshaped_15 == 0, axis=1).astype(np.int8)  # Number of clear
    single_shot_cloud_cleared_array = 15 - single_shot_cloud_cleared_array  # Number of cloudy
    name = "number_cloudy_single_shots"  # New name used here
    # pdb.set_trace()
    setattr(retv, name, np.array(single_shot_cloud_cleared_array).astype(np.int8))
    setattr(retv, "single_shot_data", np.array(data_reshaped_15).astype(np.int8))
    # We need also average cloud top and cloud base for single_shot clouds
    data = singleshotdata["ssLayer_Base_Altitude"]
    data = np.array(data)
    data_reshaped_5 = data.reshape(-1, 5)
    base_array = data_reshaped_5[:, 0]
    base_array = base_array.reshape(-1, 15)
    base_array = np.where(base_array > 0, base_array, 0.0)
    base_mean = np.where(single_shot_cloud_cleared_array > 0,
                         np.divide(np.sum(base_array, axis=1), single_shot_cloud_cleared_array),
                         - 9.0)  # Calculate average cloud base
    name = "average_cloud_base_single_shots"
    setattr(retv, name, base_mean.astype(np.float32))

    data = singleshotdata["ssLayer_Top_Pressure"]
    data = np.array(data)
    data_reshaped_5 = data.reshape(-1, 5)
    top_array = data_reshaped_5[:, 0]
    top_array = top_array.reshape(-1, 15)
    top_array = np.where(top_array > 0, top_array, 0.0)
    top_mean = np.where(single_shot_cloud_cleared_array > 0,
                        np.divide(np.sum(top_array, axis=1), single_shot_cloud_cleared_array),
                        - 9.0)  # Calculate average cloud top
    name = "average_cloud_top_pressure_single_shots"
    setattr(retv, name, top_mean.astype(np.float32))

    data = singleshotdata["ssLayer_Top_Altitude"]
    data = np.array(data)
    data_reshaped_5 = data.reshape(-1, 5)
    top_array = data_reshaped_5[:, 0]
    top_array = top_array.reshape(-1, 15)
    top_array = np.where(top_array > 0, top_array, 0.0)
    top_mean = np.where(single_shot_cloud_cleared_array > 0,
                        np.divide(np.sum(top_array, axis=1), single_shot_cloud_cleared_array),
                        - 9.0)  # Calculate average cloud top
    name = "average_cloud_top_single_shots"
    # pdb.set_trace()
    setattr(retv, name, top_mean.astype(np.float32))
    # extract less information for 1km matching
    if False:#config.RESOLUTION == 1:
        # create 1km  singel shot cfc
        data = singleshotdata["ssNumber_Layers_Found"]
        data = np.array(data).reshape(-1, 3)
        data_reshaped_3 = data.reshape(-1, 3)
        single_shot_num_clear_array = np.sum(data_reshaped_3 == 0, axis=1)  # Number of clear
        cfc_single_shots = (3 - single_shot_num_clear_array) / 3.0  # Number of cloudy
        name = "cfc_single_shots_1km_from_5km_file"  # New name used here
        # pdb.set_trace()
        setattr(retv, name, cfc_single_shots.astype(np.float32))
    return retv



def read_calipso_h5(filename, retv):
    if filename is not None:
        h5file = h5py.File(filename, 'r')
        if "Single_Shot_Detection" in h5file.keys():
            # Extract number of cloudy single shots (max 15)
            # plus average cloud base and top
            # in 5 km FOV
            print("Reading single shot information")
            retv = rearrange_calipso_the_single_shot_info(
                retv,
                {"ssNumber_Layers_Found": h5file["Single_Shot_Detection/ssNumber_Layers_Found"].value,
                 "ssLayer_Base_Altitude": h5file["Single_Shot_Detection/ssLayer_Base_Altitude"].value,
                 "ssLayer_Top_Pressure": h5file["Single_Shot_Detection/ssLayer_Top_Pressure"].value,
                 "ssLayer_Top_Altitude": h5file["Single_Shot_Detection/ssLayer_Top_Altitude"].value})
        for dataset in h5file.keys():
            if dataset in ["Single_Shot_Detection"]:
                # Handeled above
                continue
            if dataset in ["Lidar_Surface_Detection",  # New group V4
                           "metadata_t",
                           "metadata"]:
                # skip all in these groups
                print("Not reading " + dataset)
                continue
            if dataset in scip_these_larger_variables_until_needed.keys():
                print("Not reading " + dataset)
                continue
            name = dataset.lower()
            if dataset in atrain_match_names.keys():
                name = atrain_match_names[dataset]
            try:
                data = h5file[dataset].value
            except AttributeError:
                data = h5file[dataset][:]
                
            data = np.array(data)
            setattr(retv, name, data)
        h5file.close()
    return retv


def read_calipso(filename):
    print("Reading file %s", filename)
    retv = CalipsoObject()
    if filename is not None:
#         if "hdf" in filename:
#             retv = read_calipso_hdf4(filename, retv)
#         else:
        retv = read_calipso_h5(filename, retv)
    # Adopt some variables
    dsec = time.mktime((1993, 1, 1, 0, 0, 0, 0, 0, 0)) - time.timezone
    dt = datetime(1993, 1, 1, 0, 0, 0) - datetime(1970, 1, 1, 0, 0, 0)
    dsec2 = dt.days * 24 * 60 * 60
    if dsec != dsec2:
        print("WARNING")
    # 1km
#     if retv.profile_time_tai.shape == retv.number_layers_found.shape:
#         retv.latitude = retv.latitude[:, 0]
#         retv.longitude = retv.longitude[:, 0]
#         retv.profile_time_tai = retv.profile_time_tai[:, 0]
#         # Elevation is given in km's. Convert to meters:
#         retv.elevation = retv.dem_surface_elevation[:, 0] * 1000.0
#     # 5km
#     else:
    retv.latitude = retv.latitude[:, 1]
    retv.longitude = retv.longitude[:, 1]
    retv.profile_time_tai = retv.profile_time_tai[:, 1]
    # Elevation is given in km's. Convert to meters:
#     retv.elevation = retv.dem_surface_elevation[:, 2] * 1000.0
    setattr(retv, "sec_1970", retv.profile_time_tai + dsec)
    return retv






if __name__ == '__main__':
    fname = '/home/ejohansson/Scratch/Data/Calipso/CAL_LID_L2_05kmCPro-Standard-V4-20/2020/02/CAL_LID_L2_05kmCPro-Standard-V4-20.2020-02-14T09-44-07ZN.h5'
    fname = '/home/ejohansson/Scratch/Data/Calipso/CAL_LID_L2_05kmCLay.v4.20/2020/01/CAL_LID_L2_05kmCLay-Standard-V4-20.2020-01-01T00-42-10ZD.h5'
    cf = read_calipso(fname)
    pdb.set_trace()











