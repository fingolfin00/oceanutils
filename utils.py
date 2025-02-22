import netCDF4 as nc
import numpy as np
from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator, interpn
from scipy.ndimage import laplace
import glob, os
import datetime, dateutil
import pandas as pd
from .seawater import eos80

def get_coord_name (ds, keys):
    return next((k for k in ds.dimensions.keys() if k in keys), None)

def get_lev_coord_key (ds):
    lev_strings = set(["deptht", "elevation", "bathymetry", "nav_lev"])
    return get_coord_name(ds, lev_strings)

def get_lon_coord_key (ds):
    lon_strings = set(["lon", "longitude", "x", "nav_lon"])
    return get_coord_name(ds, lon_strings)

def get_lat_coord_key (ds):
    lat_strings = set(["lat", "latitude", "y", "nav_lat"])
    return get_coord_name(ds, lat_strings)

def get_var_name (ds, keys):
    return next((k for k in ds.variables.keys() if k in keys), None)

def get_lev_var_key (ds):
    lev_strings = set(["deptht", "nav_lev", "bathymetry"])
    return get_var_name(ds, lev_strings)

def get_lon_var_key (ds):
    lon_strings = set(["lon", "nav_lon", "longitude", "x"])
    return get_var_name(ds, lon_strings)

def get_lat_var_key (ds):
    lat_strings = set(["lat", "nav_lat", "latitude", "y"])
    return get_var_name(ds, lat_strings)

def create_latspace (ds):
    lat_key = get_lat_var_key(ds)
    if len(np.shape(ds.variables[lat_key])) == 2:
        return np.linspace(ds.variables[lat_key][0,0], ds.variables[lat_key][-1,0], np.shape(ds.variables[lat_key])[0])
    elif len(np.shape(ds.variables[lat_key])) == 1:
        return np.linspace(ds.variables[lat_key][0], ds.variables[lat_key][-1], np.shape(ds.variables[lat_key])[0])
    else:
        print("Lat format not supported.")
        return None

def create_lonspace (ds):
    lon_key = get_lon_var_key(ds)
    if len(np.shape(ds.variables[lon_key])) == 2:
        return np.linspace(ds.variables[lon_key][0,0], ds.variables[lon_key][0,-1], np.shape(ds.variables[lon_key])[1])
    elif len(np.shape(ds.variables[lon_key])) == 1:
        return np.linspace(ds.variables[lon_key][0], ds.variables[lon_key][-1], np.shape(ds.variables[lon_key])[0])
    else:
        print("Lon format not supported.")
        return None

def create_levspace (ds):
    lev_key = get_lev_var_key(ds)   
    return np.linspace(ds.variables[lev_key][0], ds.variables[lev_key][-1], np.shape(ds.variables[lev_key])[0])

def get_lev_from_idx (lev_i, ds):
    return create_levspace(ds)[lev_i]

def get_lon_from_idx (lon_i, ds):
    return create_lonspace(ds)[lon_i]

def get_lat_from_idx (lat_i, ds):
    return create_latspace(ds)[lat_i]

def get_idx_from_lev (lev, ds):
    return (np.abs(create_latspace(ds) - lev)).argmin()

def get_idx_from_lat (lat, ds):
    return (np.abs(create_latspace(ds) - lat)).argmin()

def get_idx_from_lon (lon, ds):
    return (np.abs(create_lonspace(ds) - lon)).argmin()

def get_idx_in_arr (l, arr):
    return (np.abs(arr - l)).argmin()

def get_parent_lat_idx_from_child (pds, cds):
    clat_s = create_latspace(cds)
    clat0, clatf = clat_s[0], clat_s[-1]
    return get_idx_from_lat(clat0, pds), get_idx_from_lat(clatf, pds)

def get_parent_lon_idx_from_child (pds, cds):
    clon_s = create_lonspace(cds)
    clon0, clonf = clon_s[0], clon_s[-1]
    return get_idx_from_lon(clon0, pds), get_idx_from_lon(clonf, pds)

def get_parent_lev_idx_from_child (pds, cds):
    clev_s = create_levspace(cds)
    clev0, clevf = clev_s[0], clev_s[-1]
    return get_idx_from_lev(clev0, pds), get_idx_from_lev(clevf, pds)

def gen_ic_ds (ds_init, var_d, filename):
    with nc.Dataset(filename, "w", format="NETCDF4") as ds:
        ds.createDimension('deptht', len(ds_init.dimensions[get_lev_coord_key(ds_init)]))
        ds.createDimension('y', len(ds_init.dimensions[get_lat_coord_key(ds_init)]))
        ds.createDimension('x', len(ds_init.dimensions[get_lon_coord_key(ds_init)]))
        ds.createVariable('nav_lat', ds_init[get_lat_var_key(ds_init)].datatype, (ds.dimensions['y'], ds.dimensions['x']))
        ds['nav_lat'][:,:] = ds_init[get_lat_var_key(ds_init)][:,:]
        ds.createVariable('nav_lon', ds_init[get_lon_var_key(ds_init)].datatype, (ds.dimensions['y'], ds.dimensions['x']))
        ds['nav_lon'][:,:] = ds_init[get_lon_var_key(ds_init)][:,:]
        ds.createVariable('deptht', ds_init[get_lev_var_key(ds_init)].datatype, (ds.dimensions['deptht']))
        ds['deptht'][:] = ds_init[get_lev_var_key(ds_init)][:]
        for k,v in var_d.items():
            ds.createVariable(k, "double", (ds.dimensions['deptht'], ds.dimensions['y'], ds.dimensions['x']))
            ds[k][:,:,:] = v

def inpaint_nan (var, kernel, timeout=2000):
    nans = np.isnan(var)
    while np.sum(nans)>0 and timeout>0:
        var[nans] = 0
        vNeighbors = convolve2d((nans==False), kernel, mode='same', boundary='symm')
        var2 = convolve2d(var, kernel, mode='same', boundary='symm')
        var2[vNeighbors>0] = var2[vNeighbors>0]/vNeighbors[vNeighbors>0]
        var2[vNeighbors==0] = np.nan
        var2[(nans==False)] = var[(nans==False)]
        var = var2
        nans = np.isnan(var)
        timeout -= 1
        if timeout==0:
            print("  Timeout!")
        # print(f"Nans: {np.size(nans)}")
    return var

def inpaint_nan_all_levels (var, levspace, kernel=[[1,1,1],[1,0,1],[1,1,1]], save_fn=None, level_timeout=2000):
    kernel_np = np.array(kernel)
    var_inpaint = var.copy()
    for l in np.arange(len(levspace)):
        print(f"--- Level: {l}")
        var_inpaint[l,:,:] = inpaint_nan(var_inpaint[l,:,:], kernel=kernel_np, timeout=level_timeout)
    print(f"var inpainted.")
    if save_fn:
        print(f"save in file {save_fn}")
        np.save(save_fn, np.asarray(var_inpaint))
    return var_inpaint


def fill_zeros (var):    
    for i in np.arange(np.shape(var)[2]): # lon
        for j in np.arange(np.shape(var)[1]): #lat
            col = var[:,j,i]
            np.nan_to_num(col, copy=False, nan=col[np.isnan(np.roll(col,-1,axis=0))][0])
    # crazy onliner not sure why it works
    # np.nan_to_num(var, copy=False, nan=np.reshape(var[np.isnan(np.roll(var,-1,axis=0))], np.shape(var)[1:]))
    return var

def interpolate_parent_in_child (pvar, pds, cds, pad=0, mode='2d', method='linear'):
    clat_s, clon_s, clev_s = create_latspace(cds), create_lonspace(cds), create_levspace(cds)
    plat_s, plon_s, plev_s = create_latspace(pds), create_lonspace(pds), create_levspace(pds)
    plat0_i, platf_i = get_parent_lat_idx_from_child(pds, cds)
    plon0_i, plonf_i = get_parent_lon_idx_from_child(pds, cds)
    cvar = np.empty((len(clev_s), len(clat_s), len(clon_s)))
    # print("Parent: ", np.shape(plat_s[plat0_i:platf_i]), np.shape(plon_s[plon0_i:plonf_i]))
    # print(plon_s[plon0_i:plonf_i][0], plon_s[plon0_i:plonf_i][-1])
    # print(plat_s[plat0_i:platf_i][0], plat_s[plat0_i:platf_i][-1])
    # print("Var: ", np.shape(pvar[0,plat0_i:platf_i,plon0_i:plonf_i]))
    # print("Child: ", np.shape(cvar))
    if mode=='2d' and len(clev_s) == len(plev_s):
        for l in np.arange(len(plev_s[:])):
            bi_cvar = RegularGridInterpolator([plat_s[plat0_i-pad:platf_i+pad], plon_s[plon0_i-pad:plonf_i+pad]], pvar[l,plat0_i-pad:platf_i+pad,plon0_i-pad:plonf_i+pad], method=method)
            CLON, CLAT = np.meshgrid(clon_s, clat_s)
            # print("Child grid: ", np.shape(CLON), np.shape(CLAT))
            # print(CLON[0,0], CLON[0,-1], CLAT[0,0], CLAT[-1,0])
            cvar[l,:,:] = bi_cvar((CLAT, CLON))
        return cvar
    elif mode=='3d':
        plev0_i, plevf_i = get_parent_lev_idx_from_child(pds, cds) 
        tri_cvar = interpn([plev_s[plev0_i:plevf_i], plat_s[plat0_i:platf_i], plon_s[plon0_i:plonf_i]], cvar, np.meshgrid(clev_s, clat_s, clon_s, indexing='ij'), method=method)
        return tri_cvar
    else:
        print("Not a valid mode")
        return cvar

def apply_laplace (field, N, fac, mode='nearest'):
    field_filtered = field.copy()
    for i in range(N):
        field_filtered += laplace(field_filtered, mode=mode)/fac
    return field_filtered

def broadcast_lev_3d (lev_1d, lat_dim, lon_dim):
    return np.repeat([np.repeat([lev_1d], lat_dim, axis=0)], lon_dim, axis=0).T

def broadcast_lat_3d (lat_1d, lev_dim, lon_dim):
    return np.repeat([np.repeat([lat_1d], lev_dim, axis=0).T], lon_dim, axis=0).T

def pressure (ds):
    return eos80.pres(broadcast_lev_3d(create_levspace(ds)[:], len(create_latspace(ds)), len(create_lonspace(ds))), broadcast_lat_3d(create_latspace(ds), len(create_levspace(ds)), len(create_lonspace(ds))))

def potdensity (vot, vos, pres):
    return eos80.dens(vos, eos80.T90conv(vot), pres) # kg/m^3

def grad_dz (field):
    return np.append(np.diff(field, 1, axis=0), [np.zeros(np.shape(field)[1:])], axis=0) # last level all zeros to keep dimensions consistent

def move_down_neg_potdensity (vot, vos, potd, dpotd_bool, pres, mask):
    # solo nella T al k+1 meno un decimo di grado, reiterare fino a che non ho più diff densità pot sotto zero
    # se il problema peggiora aggiungere un centesimo di PSU nella salinità
    vot_filtered, vos_filtered, potd_filtered, dpotd_bool_filtered = vot.copy(), vos.copy(), potd.copy(), dpotd_bool.copy()
    dpotd_dz = grad_dz(potd_filtered)
    print(dpotd_bool_filtered.sum())
    while dpotd_bool_filtered.sum()>0:
        rolled_dpotd_bool = np.roll(dpotd_bool_filtered,1,axis=0)
        rolled_dpotd_bool[0,:,:] = np.full(np.shape(rolled_dpotd_bool)[1:], False)
        # vot_filtered[rolled_dpotd_bool] = vot_filtered[dpotd_bool_filtered]
        vot_filtered[rolled_dpotd_bool] -= 0.1
        vos[rolled_dpotd_bool] += 0.01
        potd_filtered[rolled_dpotd_bool] = eos80.dens(vos_filtered[rolled_dpotd_bool], eos80.T90conv(vot_filtered[rolled_dpotd_bool]), pres[rolled_dpotd_bool])
        dpotd_dz = np.append(np.diff(potd_filtered, 1, axis=0), [np.zeros(np.shape(potd_filtered)[1:])], axis=0)
        dpotd_bool_filtered = np.asarray(dpotd_dz*mask<0)
        print(dpotd_bool_filtered.sum())

    return vot_filtered, vos_filtered, potd_filtered, dpotd_dz

def get_zonal_outflow (u, e2u, e3u, units='sverdrup'):
    mask = u<0
    doutflow = u*e2u*e3u*mask
    if units == 'sverdrup':
        outflow = np.sum(doutflow, axis=(1,2,3))/10**6
    else:
        outflow = np.sum(doutflow, axis=(1,2,3))
    return outflow

def get_zonal_inflow (u, e2u, e3u, units='sverdrup'):
    mask = u>0
    dinflow = u*e2u*e3u*mask
    if units == 'sverdrup':
        inflow = np.sum(dinflow, axis=(1,2,3))/10**6
    else:
        inflow = np.sum(dinflow, axis=(1,2,3))
    return inflow

def rolling_mean (field, window):
    return np.convolve(np.ravel(np.array(field)), np.ones(window)/window, mode='valid')

def get_restart_strformat (restart_freq):
    if 'd' in restart_freq:
        return "%Y%m%d"
    elif 'm' in restart_freq: # untested
        return "%Y%m"
    else:
        print(f"Restart freq {restart_freq} not supported")
        return None

def get_ncfiles (exp_name, start_date, end_date, workpath, recursive=False, datapoint_freq='6h', restart_freq='15d', save_freq='1m'):
    print(workpath)
    glob_star = r'/**/' if recursive else r'*/*'
    if save_freq=='1m':
        save_strformat = "%Y%m"
        save_delta = dateutil.relativedelta.relativedelta(months=+1)
    else:
        print(f"Not supported save freq format. {save_freq}")
        return
    grids = ["T", "U", "V", "W"]
    restart_strformat = get_restart_strformat(restart_freq=restart_freq)
    #restart_dates = [single_date for single_date in daterange(start_date, end_date, restart_days)]
    restart_dates = pd.date_range(start=start_date, end=end_date, freq=restart_freq).to_pydatetime().tolist()
    fn = {g: {} for g in grids}
    for i in range(len(restart_dates[:-1])):
        d_start = restart_dates[:-1][i]
        d_end = restart_dates[1:][i] - datetime.timedelta(days=1)
        # d_end = restart_dates[1:][i]
        # dates = [f"{d.strftime(save_strformat)}-{(d+save_delta).strftime(save_strformat)}" for d in pd.date_range(start=d_start, end=d_end, freq=datapoint_freq).to_pydatetime().tolist()]
        d_save = f"{d_start.strftime(save_strformat)}-{(d_start+save_delta).strftime(save_strformat)}"
        d_restart = f"{d_start.strftime(restart_strformat)}_{d_end.strftime(restart_strformat)}"
        # for d in dates:
        for g in grids:
            # search_str = f"{exp_name}_{datapoint_freq}_{d_start.strftime(restart_strformat)}_{d_end.strftime(restart_strformat)}_grid_{g}_{d_save}.nc"
            search_str = f"{exp_name}_{datapoint_freq}_{d_start.strftime(restart_strformat)}_{d_end.strftime(restart_strformat)}_grid_{g}_*.nc"
            print(search_str)
            # fn[g][d_restart] = glob.glob(workpath+r"*/"+search_str)
            fn[g][d_restart] = glob.glob(workpath+glob_star+search_str)
    return fn

def get_transport_timeseries (fn, start_date, end_date, lat0, latf, lon, dpds, restart_freq='15d'):
    grid = 'U'
    restart_strformat = get_restart_strformat(restart_freq=restart_freq)
    restart_dates = pd.date_range(start=start_date, end=end_date, freq=restart_freq).to_pydatetime().tolist()
    restart_str = [f"{restart_dates[:-1][i].strftime(restart_strformat)}_{(restart_dates[1:][i] - datetime.timedelta(days=1)).strftime(restart_strformat)}" for i in range(len(restart_dates[:-1]))]
    print(restart_str)
    lat0_i, latf_i = get_idx_from_lat(lat0,dpds), get_idx_from_lat(latf,dpds)
    lon0_i, lonf_i = get_idx_from_lon(lon,dpds), get_idx_from_lon(lon,dpds)+1
    outflow, inflow = np.array([]), np.array([])
    for s in restart_str:
        print(s)
        upfn = os.path.abspath(fn[grid][s][0])
        upds = nc.Dataset(fn[grid][s][0])
        # print(upds.variables)
        # print(lat0_i, latf_i)
        # print(lon0_i, lonf_i)
        # print("get u")
        u = upds.variables['vozocrtx'][:,:,lat0_i:latf_i,lon0_i:lonf_i]
        # print(np.shape(u))
        e2u = np.repeat([np.repeat([dpds.variables['e2u'][0,lat0_i:latf_i,lon0_i:lonf_i]], len(create_levspace(dpds)), axis=0)], np.shape(u)[0], axis=0)
        e3u = np.repeat([dpds.variables['e3u_0'][0,:,lat0_i:latf_i,lon0_i:lonf_i]], np.shape(u)[0], axis=0)
        # print("Calc outflow")
        outflow = np.append(outflow, get_zonal_outflow(u, e2u, e3u))
        # print("Calc inflow")
        inflow = np.append(inflow, get_zonal_inflow(u, e2u, e3u))
        net = outflow + inflow
    return outflow, inflow, net

def get_full_period (varname, fnd, start_date, end_date, restart_strformat):
    print(list(fnd.keys())[0])
    var = nc.Dataset(fnd[list(fnd.keys())[0]][0]).variables[varname][:,:,:,:]
    for k,v in fnd.items():
        if start_date.strftime(restart_strformat) not in k:
            print(k)
            np.append(var, nc.Dataset(v[0]).variables[varname][:,:,:,:])
            if end_date.strftime(restart_strformat) in k:
                return var 

def savez_data (data, data_str, start_date, end_date, restart_strformat, fld='./'):
    fn = f"numpy_data-{start_date.strftime(restart_strformat)}_{end_date.strftime(restart_strformat)}-{data_str}.npy"
    with open(f"{fld}{fn}", 'wb') as f:
        np.savez(f"{fld}{fn}", data)

def loadz_data (folder, arr_name='arr_0'):
    datad = {}
    fns = glob.glob(folder+"/*.npz")
    for fn in fns:
        # print(fn)
        with open(fn, 'rb') as f:
            datak = os.path.basename(fn).split('.')[0].split('-')[-1]
            print(datak)
            datad[datak] = np.load(f)[arr_name]
    return datad
