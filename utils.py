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
    lev_strings = set(["deptht", "depthu", "depthv", "depthw", "elevation", "bathy_metry", "bathymetry", "nav_lev"])
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
    lev_strings = set(["deptht", "depthu", "depthv", "depthw", "elevation", "nav_lev", "bathy_metry", "bathymetry"])
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
        # print(2)
        # print(ds.variables[lat_key][:,0])
        # return np.linspace(ds.variables[lat_key][0,0], ds.variables[lat_key][-1,0], np.shape(ds.variables[lat_key])[0])
        return ds.variables[lat_key][:,0]
    elif len(np.shape(ds.variables[lat_key])) == 1:
        # print(1)
        # print(ds.variables[lat_key][:])
        # return np.linspace(ds.variables[lat_key][0], ds.variables[lat_key][-1], np.shape(ds.variables[lat_key])[0])
        return ds.variables[lat_key][:]
    else:
        print("Lat format not supported.")
        return None

def create_lonspace (ds):
    lon_key = get_lon_var_key(ds)
    # print(lon_key)
    if len(np.shape(ds.variables[lon_key])) == 2:
        # print(2)
        # print(ds.variables[lon_key][40,:])
        # return np.linspace(ds.variables[lon_key][0,0], ds.variables[lon_key][0,-1], np.shape(ds.variables[lon_key])[1])
        return ds.variables[lon_key][40,:]
    elif len(np.shape(ds.variables[lon_key])) == 1:
        # print(1)
        # print(ds.variables[lon_key][:])
        # return np.linspace(ds.variables[lon_key][0], ds.variables[lon_key][-1], np.shape(ds.variables[lon_key])[0])
        return ds.variables[lon_key][:]
    else:
        print("Lon format not supported.")
        return None

def create_levspace (ds):
    lev_key = get_lev_var_key(ds)
    # variables['nav_lev'][:]
    # return np.linspace(ds.variables[lev_key][0], ds.variables[lev_key][-1], np.shape(ds.variables[lev_key])[0])
    return ds.variables[lev_key][:]

def get_lev_from_idx (lev_i, ds):
    return create_levspace(ds)[lev_i]

def get_lon_from_idx (lon_i, ds):
    return create_lonspace(ds)[lon_i]

def get_lat_from_idx (lat_i, ds):
    return create_latspace(ds)[lat_i]

def get_idx_from_lev (lev, ds):
    return (np.abs(create_levspace(ds) - lev)).argmin()

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

def get_child_coords (dcads, nghost=0):
    return ((create_levspace(dcads)[0],create_levspace(dcads)[-1]),(create_latspace(dcads)[nghost],create_latspace(dcads)[-1-nghost]),(create_lonspace(dcads)[nghost],create_lonspace(dcads)[-1-nghost]))

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

def grad (field, axis):
    return np.append(np.diff(field, 1, axis=axis), [np.zeros(np.shape(field)[1:])], axis=axis) # last level all zeros to keep dimensions consistent

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

def rolling_mean_paolo (field, window):
    out = np.ma.array([])
    for n in np.arange(np.shape(field)[0]):
        idx1 = np.max([0, n-int(window/2)])
        idx2 = np.min([np.shape(field)[0], n+int(window/2)])
        
        out = np.ma.append(out, np.ma.mean(field[idx1:idx2], axis=0))
    return out.reshape(np.shape(field))

def rolling_mean (field, window):
    return np.convolve(np.ravel(np.array(field)), np.ones(window)/window, mode='same')

def down_sample(x, f=7):
    # pad to a multiple of f, so we can reshape
    # use nan for padding, so we needn't worry about denominator in
    # last chunk
    xp = np.ma.masked_invalid(np.r_[x, np.nan + np.zeros((-np.shape(x)[0] % f,))], np.nan)
    # reshape, so each chunk gets its own row, and then take mean
    return np.ma.mean(xp.reshape(-1, f), axis=-1)

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

def get_scale_factor (domds, grid, var, zoom_coords=((None,None),(None,None),(None,None))): # lev, lat, lon
    lev0_i, levf_i = get_idx_from_lev(zoom_coords[0][0], domds) if zoom_coords[0][0] else None, get_idx_from_lev(zoom_coords[0][1], domds) if zoom_coords[0][1] else None
    lat0_i, latf_i = get_idx_from_lat(zoom_coords[1][0], domds) if zoom_coords[1][0] else None, get_idx_from_lat(zoom_coords[1][1], domds) if zoom_coords[1][1] else None
    lon0_i, lonf_i = get_idx_from_lon(zoom_coords[2][0], domds) if zoom_coords[2][0] else None, get_idx_from_lon(zoom_coords[2][1], domds) if zoom_coords[2][1] else None
    # print(lev0_i, levf_i, lat0_i, latf_i, lon0_i, lonf_i)
    e1 = np.repeat([np.repeat([domds.variables['e1'+grid.lower()][0,lat0_i:latf_i,lon0_i:lonf_i]], len(create_levspace(domds)[lev0_i:levf_i]), axis=0)], np.shape(var)[0], axis=0)
    e2 = np.repeat([np.repeat([domds.variables['e2'+grid.lower()][0,lat0_i:latf_i,lon0_i:lonf_i]], len(create_levspace(domds)[lev0_i:levf_i]), axis=0)], np.shape(var)[0], axis=0)
    e3 = np.repeat([domds.variables['e3'+grid.lower()+'_0'][0,lev0_i:levf_i,lat0_i:latf_i,lon0_i:lonf_i]], np.shape(var)[0], axis=0)
    return e1, e2, e3

def get_var_from_ds (varname, ds, domds, grid, zoom_coords=((None,None),(None,None),(None,None))): # lev, lat, lon
    lev0_i, levf_i = get_idx_from_lev(zoom_coords[0][0], domds) if zoom_coords[0][0] else None, get_idx_from_lev(zoom_coords[0][1], domds) if zoom_coords[0][1] else None
    lat0_i, latf_i = get_idx_from_lat(zoom_coords[1][0], domds) if zoom_coords[1][0] else None, get_idx_from_lat(zoom_coords[1][1], domds) if zoom_coords[1][1] else None
    lon0_i, lonf_i = get_idx_from_lon(zoom_coords[2][0], domds) if zoom_coords[2][0] else None, get_idx_from_lon(zoom_coords[2][1], domds) if zoom_coords[2][1] else None
    # print(lev0_i, levf_i, lat0_i, latf_i, lon0_i, lonf_i)
    var = ds.variables[varname][:,lev0_i:levf_i,lat0_i:latf_i,lon0_i:lonf_i]
    return var

def get_ke (u, v):
    return 0.5*(u**2 + v**2)

def get_ke_3dspatial_mean (u, v, e1u, e2u, e3u, e1v, e2v, e3v):
    us = np.ma.sum(0.5*np.ma.power(u,2)*e1u*e2u*e3u, axis=(1,2,3)) / np.ma.sum(e1u*e2u*e3u, axis=(1,2,3))
    vs = np.ma.sum(0.5*np.ma.power(v,2)*e1v*e2v*e3v, axis=(1,2,3)) / np.ma.sum(e1v*e2v*e3v, axis=(1,2,3))
    return us+vs

def get_ke_2dspatial_mean (u, v, e1u, e2u, e1v, e2v):
    us = np.ma.sum(0.5*np.ma.power(u,2)*e1u*e2u, axis=(2,3)) / np.ma.sum(e1u*e2u, axis=(2,3))
    vs = np.ma.sum(0.5*np.ma.power(v,2)*e1v*e2v, axis=(2,3)) / np.ma.sum(e1v*e2v, axis=(2,3))
    return us+vs

def get_meridional_outflow (v, e1v, e3v, units='sverdrup'):
    mask = v<0
    doutflow = v*e1v*e3v*mask
    if units == 'sverdrup':
        outflow = np.ma.sum(doutflow, axis=(1,2,3))/10**6
    else:
        outflow = np.ma.sum(doutflow, axis=(1,2,3))
    # print(outflow)
    return outflow

def get_meridional_inflow (v, e1v, e3v, units='sverdrup'):
    mask = v>0
    dinflow = v*e1v*e3v*mask
    if units == 'sverdrup':
        inflow = np.ma.sum(dinflow, axis=(1,2,3))/10**6
    else:
        inflow = np.ma.sum(dinflow, axis=(1,2,3))
    return inflow

def get_zonal_outflow (u, e2u, e3u, units='sverdrup'):
    mask = u<0
    doutflow = u*e2u*e3u*mask
    if units == 'sverdrup':
        outflow = np.ma.sum(doutflow, axis=(1,2,3))/10**6
    else:
        outflow = np.ma.sum(doutflow, axis=(1,2,3))
    # print(outflow)
    return outflow

def get_zonal_inflow (u, e2u, e3u, units='sverdrup'):
    mask = u>0
    dinflow = u*e2u*e3u*mask
    if units == 'sverdrup':
        inflow = np.ma.sum(dinflow, axis=(1,2,3))/10**6
    else:
        inflow = np.ma.sum(dinflow, axis=(1,2,3))
    return inflow

def get_zonal_inflow_tracer (u, tracer, e2t, e3t):
    mask = u>0
    return np.ma.sum(tracer*e2t*e3t, axis=(1,2,3)) / np.ma.sum(e2t*e3t, axis=(1,2,3))

def get_zonal_outlow_tracer (u, tracer, e2t, e3t):
    mask = u<0
    return np.ma.sum(tracer*e2t*e3t, axis=(1,2,3)) / np.ma.sum(e2t*e3t, axis=(1,2,3))

def get_change_sign (arr):
    return (np.ma.diff(np.sign(arr)) > 0)*1 + (np.ma.diff(np.sign(arr)) < 0)*(-1)

def get_interface (u, domds):
    mean_profile = np.ma.mean(u[:,:,:,:], axis=(2,3))
    cs_mean_profile = get_change_sign(mean_profile)
    interface = np.array([])
    for i in np.arange(np.shape(cs_mean_profile)[0]):
        interface_idxs = np.where(cs_mean_profile[i]==-1)
        # print(interface_idxs)
        if len(interface_idxs[0]) > 0:
            interface = np.append(interface, create_levspace(domds)[interface_idxs[0][0]])
        else:
            interface = np.append(interface, np.nan)
    return mean_profile, np.ma.masked_invalid(interface, np.nan)

def get_interface_old (u, domds, interface_minlev=400):
    mean_profile = np.ma.mean(u[:,:,:,:], axis=(2,3))
    interface_minlev_idx = get_idx_from_lev(interface_minlev, domds)
    interface = np.ma.array([])
    z = create_levspace(domds)[0:interface_minlev_idx]
    cut_mean_profile = mean_profile[:, 0:interface_minlev_idx]
    for i in range(np.shape(cut_mean_profile)[0]):
        interface_idx = get_idx_in_arr(0, cut_mean_profile[i,:])
        interface = np.ma.append(interface, z[interface_idx])
    return mean_profile, interface

def get_full_interface (fnds, domds, start_date, end_date, full_start_date, full_end_date,
                   lat0, latf, lon, restart_freq='15d'):
    
    lat0_i, latf_i = get_idx_from_lat(lat0,domds), get_idx_from_lat(latf,domds)
    lon0_i, lonf_i = get_idx_from_lon(lon,domds), get_idx_from_lon(lon,domds)+1
    lon0, lonf = get_lon_from_idx(lon0_i,domds), get_lon_from_idx(lonf_i,domds)
    # print(lat0_i, latf_i, lon0_i, lonf_i)
    zoom_coords = ((None,None),(lat0, latf),(lon0, lonf))

    u = get_full_period ('vozocrtx', fnds['U'], domds, start_date, end_date, full_start_date, full_end_date,
                         zoom_coords=zoom_coords) # lev, lat, lon
    _, e2u, e3u = get_scale_factor(domds, 'U', u, zoom_coords=zoom_coords)
    return get_interface(u, domds)

def get_full_tracer_flow (fnds, domds, start_date, end_date, full_start_date, full_end_date,
                   lat0, latf, lon, restart_freq='15d'):
    lat0_i, latf_i = get_idx_from_lat(lat0,domds), get_idx_from_lat(latf,domds)
    lon0_i, lonf_i = get_idx_from_lon(lon,domds), get_idx_from_lon(lon,domds)+1
    lon0, lonf = get_lon_from_idx(lon0_i,domds), get_lon_from_idx(lonf_i,domds)
    # print(lat0_i, latf_i, lon0_i, lonf_i)
    zoom_coords = ((None,None),(lat0, latf),(lon0, lonf))

    T = get_full_period ('votemper', fnds['T'], domds, start_date, end_date, full_start_date, full_end_date,
                     zoom_coords=zoom_coords) # lev, lat, lon
    S = get_full_period ('vosaline', fnds['T'], domds, start_date, end_date, full_start_date, full_end_date,
                     zoom_coords=zoom_coords)
    _, e2t, e3t = get_scale_factor(domds, 'T', u, zoom_coords=zoom_coords)

    zonalinT = get_zonal_inflow_tracer(u, T, e2t, e3t)
    zonalinS = get_zonal_inflow_tracer(u, S, e2t, e3t)
    zonaloutT = get_zonal_outlow_tracer(u, T, e2t, e3t)
    zonaloutS = get_zonal_outlow_tracer(u, S, e2t, e3t)

    return zonalinT, zonalinS, zonaloutT, zonaloutS

def get_zonal_metrics (fnds, domds, start_date, end_date, full_start_date, full_end_date,
                       lat0, latf, lon, restart_freq='15d'):
    
    lat0_i, latf_i = get_idx_from_lat(lat0,domds), get_idx_from_lat(latf,domds)
    lon0_i, lonf_i = get_idx_from_lon(lon,domds), get_idx_from_lon(lon,domds)+1
    lon0, lonf = get_lon_from_idx(lon0_i,domds), get_lon_from_idx(lonf_i,domds)
    # print(lat0_i, latf_i, lon0_i, lonf_i)
    zoom_coords = ((None,None),(lat0, latf),(lon0, lonf))

    u = get_full_period ('vozocrtx', fnds['U'], domds, start_date, end_date, full_start_date, full_end_date,
                     zoom_coords=zoom_coords) # lev, lat, lon
    _, e2u, e3u = get_scale_factor(domds, 'U', u, zoom_coords=zoom_coords)
    
    T = get_full_period ('votemper', fnds['T'], domds, start_date, end_date, full_start_date, full_end_date,
                     zoom_coords=zoom_coords) # lev, lat, lon
    S = get_full_period ('vosaline', fnds['T'], domds, start_date, end_date, full_start_date, full_end_date,
                     zoom_coords=zoom_coords)
    _, e2t, e3t = get_scale_factor(domds, 'T', u, zoom_coords=zoom_coords)
    
    outflow = get_zonal_outflow(u, e2u, e3u)
    inflow = get_zonal_inflow(u, e2u, e3u)
    netflow = outflow + inflow

    zonalinT = get_zonal_inflow_tracer(u, T, e2t, e3t)
    zonalinS = get_zonal_inflow_tracer(u, S, e2t, e3t)
    zonaloutT = get_zonal_outlow_tracer(u, T, e2t, e3t)
    zonaloutS = get_zonal_outlow_tracer(u, S, e2t, e3t)

    mean_profile, interface = get_interface(u, domds)
    
    return {'outflow': outflow, 'inflow': inflow, 'netflow': netflow,
            'zonalinT': zonalinT, 'zonalinS': zonalinS, 'zonaloutT': zonaloutT, 'zonaloutS': zonaloutS, 
            'interface': interface, 'mean_profile': mean_profile}

def get_meridional_transport (fnds, domds, start_date, end_date, full_start_date, full_end_date,
                       lat0, latf, lon, restart_freq='15d'):
    lat0_i, latf_i = get_idx_from_lat(lat0,domds), get_idx_from_lat(latf,domds)
    lon0_i, lonf_i = get_idx_from_lon(lon,domds), get_idx_from_lon(lon,domds)+1
    lon0, lonf = get_lon_from_idx(lon0_i,domds), get_lon_from_idx(lonf_i,domds)
    # print(lat0_i, latf_i, lon0_i, lonf_i)
    zoom_coords = ((None,None),(lat0, latf),(lon0, lonf))

    v = get_full_period ('vomecrty', fnds['V'], domds, start_date, end_date, full_start_date, full_end_date,
                     zoom_coords=zoom_coords) # lev, lat, lon
    e1v, _, e3v = get_scale_factor(domds, 'V', v, zoom_coords=zoom_coords)

    outflow = get_meridional_outflow(v, e1v, e3v)
    inflow = get_meridional_inflow(v, e1v, e3v)
    netflow = outflow + inflow
    return {'mer_outflow': outflow, 'mer_inflow': inflow, 'mer_netflow': netflow}

def from_date_to_datetime (date_datefmt):
    return datetime.datetime.combine(date_datefmt, datetime.datetime.min.time())

def from_date_to_idx (start_date, end_date, freq='6h'):
    # print(start_date, end_date)
    daterange = pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()
    # print(daterange)
    return daterange.index(daterange[-1]) 

def get_full_period (varname, fnds, domds,
                     start_date, end_date, full_start_date, full_end_date,
                     vardim='3d', restart_freq='15d', save_freq='6h',
                     zoom_coords=((None,None),(None,None),(None,None))): # lev, lat, lon
    full_restart_dates = pd.date_range(start=full_start_date, end=full_end_date, freq=restart_freq).to_pydatetime().tolist()
    start_date, end_date = start_date, end_date
    full_start_date, full_end_date = full_start_date, full_end_date
    if full_end_date > full_restart_dates[-1]:
        full_restart_dates.append(full_end_date+datetime.timedelta(days=1))
    lev0_i, levf_i = get_idx_from_lev(zoom_coords[0][0], domds) if zoom_coords[0][0] else None, get_idx_from_lev(zoom_coords[0][1], domds) if zoom_coords[0][1] else None
    lat0_i, latf_i = get_idx_from_lat(zoom_coords[1][0], domds) if zoom_coords[1][0] else None, get_idx_from_lat(zoom_coords[1][1], domds) if zoom_coords[1][1] else None
    lon0_i, lonf_i = get_idx_from_lon(zoom_coords[2][0], domds) if zoom_coords[2][0] else None, get_idx_from_lon(zoom_coords[2][1], domds) if zoom_coords[2][1] else None
    print(f"Load {varname} in ", lev0_i, levf_i, lat0_i, latf_i, lon0_i, lonf_i)
    # print(full_restart_dates)
    # print(list(fnds.keys()))
    var = None
    if start_date >= full_start_date or end_date <= full_end_date or start_date < end_date:
        for k,v in fnds.items():
            i = list(fnds.keys()).index(k)+1
            # print(k, i, v, full_restart_dates[i])
            if start_date <= full_restart_dates[i]:
                # print(k, full_restart_dates[i])
                # print(v[0])
                ds = nc.Dataset(v[0])
                # print(lev0_i, levf_i, lat0_i, latf_i, lon0_i, lonf_i)
                # print(zoom_coords[2][0], zoom_coords[2][1])
                if var is not None:
                    if end_date <= full_restart_dates[i]:
                        # print(full_restart_dates[i-1], end_date)
                        var_end_idx = from_date_to_idx(full_restart_dates[i-1], end_date, freq=save_freq)+1 if end_date < full_restart_dates[i] else None
                        print('end idx: ',var_end_idx)
                        # print(end_date)
                        # print(i, k, "append last var")
                        if vardim == '3d':
                            var = np.ma.append(var, ds.variables[varname][:var_end_idx,lev0_i:levf_i,lat0_i:latf_i,lon0_i:lonf_i], axis=0)
                        elif vardim == '2d':
                            var = np.ma.append(var, ds.variables[varname][:var_end_idx,lat0_i:latf_i,lon0_i:lonf_i], axis=0)
                        else:
                            print(f"Var dim {vardim} not supported.")
                        # print(np.shape(var))
                        return var
                    else:
                        # print(i, k, f"append var")
                        if vardim == '3d':
                            var = np.ma.append(var, ds.variables[varname][:,lev0_i:levf_i,lat0_i:latf_i,lon0_i:lonf_i], axis=0)
                        elif vardim == '2d':
                            var = np.ma.append(var, ds.variables[varname][:,lat0_i:latf_i,lon0_i:lonf_i], axis=0)
                        else:
                            print(f"Var dim {vardim} not supported.")
                        
                        # print(np.shape(var))
                else:
                    # print(full_restart_dates[i-1])
                    # print(full_restart_dates[i])
                    var_start_idx = from_date_to_idx(full_restart_dates[i-1], start_date, freq=save_freq) if start_date < full_restart_dates[i] else None
                    # print('start date:',start_date)
                    print('start idx:',var_start_idx)
                    # handle interval within a single ds
                    if end_date <= full_restart_dates[i]:
                        var_end_idx = from_date_to_idx(full_restart_dates[i-1], end_date, freq=save_freq)+1 if end_date < full_restart_dates[i] else None
                        print('end idx:',var_end_idx)
                        # print('end date:',end_date)
                        # print(i, k, "single ds")
                        if vardim == '3d':
                            var = ds.variables[varname][var_start_idx:var_end_idx,lev0_i:levf_i,lat0_i:latf_i,lon0_i:lonf_i]
                        elif vardim == '2d':
                            var = ds.variables[varname][var_start_idx:var_end_idx,lat0_i:latf_i,lon0_i:lonf_i]
                        else:
                            print(f"Var dim {vardim} not supported.")
                        
                        # print(np.shape(var))
                        return var
                    else:
                        # create first var
                        print(i, k, "first var")
                        if vardim == '3d':
                            var = ds.variables[varname][var_start_idx:,lev0_i:levf_i,lat0_i:latf_i,lon0_i:lonf_i]
                        elif vardim == '2d':
                            var = ds.variables[varname][var_start_idx:,lat0_i:latf_i,lon0_i:lonf_i]
                        else:
                            print(f"Var dim {vardim} not supported.")
                        
                        # print(np.shape(var))
            else:
                continue
    else:
        print("Invalid interval")
        return None

def get_means (varname, fnds, domds, freq, full_start_date, full_end_date, start_date, end_date,
                       zoom_coords=((None,None),(None,None),(None,None)), include_zeroh_boundary=False, vardim='3d'):
    print(f"Get {freq}-mean for {varname}, {start_date}-{end_date}")
    range_end = pd.date_range(start=start_date, end=end_date, freq=freq).to_pydatetime().tolist()
    delta_range = range_end[1]-range_end[0]
    range_start = pd.date_range(start=start_date-delta_range, end=end_date-delta_range, freq=freq).to_pydatetime().tolist()
    var_mean = None
    for s,e in zip(range_start, range_end):
        start = s+datetime.timedelta(days=1)
        end = e+datetime.timedelta(hours=24) if include_zeroh_boundary else e+datetime.timedelta(hours=18)
        print(start, end)
        full_period = get_full_period(varname, fnds, domds, start, end, full_start_date, full_end_date, zoom_coords=zoom_coords, vardim=vardim)
        if var_mean is None:
            var_mean = np.ma.array([np.ma.mean(full_period, axis=0)])
        else:
            var_mean = np.ma.append(var_mean, np.ma.array([np.ma.mean(full_period, axis=0)]), axis=0)
    return var_mean

def get_restart_strformat (restart_freq):
    if 'd' in restart_freq:
        return "%Y%m%d"
    elif 'm' in restart_freq: # untested
        return "%Y%m"
    else:
        print(f"Restart freq {restart_freq} not supported")
        return None

def get_ncfiles (exp_name, start_date, end_date, workpath, recursive=False, datapoint_freq='6h', restart_freq='15d', save_freq='1m'):
    # print(workpath)
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
            search_str = f"{exp_name}_{datapoint_freq}_{d_restart}_grid_{g}_*.nc"
            # print(search_str)
            # fn[g][d_restart] = glob.glob(workpath+r"*/"+search_str)
            fn[g][d_restart] = glob.glob(workpath+glob_star+search_str)
    last_d_end = datetime.datetime.combine(end_date, datetime.datetime.min.time()) - datetime.timedelta(days=1)
    if last_d_end > restart_dates[-1]:
        last_restart = f"{restart_dates[-1].strftime(restart_strformat)}_{last_d_end.strftime(restart_strformat)}"
        for g in grids:
            last_search_str = f"{exp_name}_{datapoint_freq}_{last_restart}_grid_{g}_*.nc"
            fn[g][last_restart] = glob.glob(workpath+glob_star+last_search_str)
    return fn

def savez_data (data, data_str, start_date, end_date, restart_freq='15d', fld='./', masked=True):
    restart_strformat = get_restart_strformat(restart_freq=restart_freq)
    fn = f"{start_date.strftime(restart_strformat)}_{end_date.strftime(restart_strformat)}-{data_str}.npy.npz"
    os.makedirs(fld, exist_ok=True)
    with open(f"{fld}{fn}", 'wb') as f:
        print(f"Saving {data_str} in {fld}{fn}")
        np.savez(f"{fld}{fn}", np.ma.filled(data, np.nan) if masked else data)

def loadz_data (varname, folder, start_date, end_date, restart_freq='15d', arr_name='arr_0', masked=True, slices=(None,None,None,None)):
    time, lev, lat, lon = slices[0], slices[1], slices[2], slices[3]
    print(time, lev)
    restart_strformat = get_restart_strformat(restart_freq=restart_freq)
    datad = {}
    fns = glob.glob(folder+f"/{start_date.strftime(restart_strformat)}_{end_date.strftime(restart_strformat)}-*{varname}*.npz")
    for fn in fns:
        # print(fn)
        with open(fn, 'rb') as f:
            datak = os.path.basename(fn).split('.')[0].split('-')[-1]
            print(datak)
            if lev and time:
                datad[datak] = np.ma.masked_invalid(np.load(f)[arr_name][time,lev,:,:]) if masked else np.load(f)[arr_name][time,lev,:,:]
            else:
                datad[datak] = np.ma.masked_invalid(np.load(f)[arr_name]) if masked else np.load(f)[arr_name]
    return datad

def savez_mean_fields (savez_fld, varname, fn_d, grid, freq, full_start_date, full_end_date, start_date, end_date,
                      restart_freq='15d', zoom_coords=((None,None),(None,None),(None,None)), vardim='3d', masked=True): # lev, lat, lon

    os.makedirs(savez_fld, exist_ok=True)
    means_d = {}
    for dsname, (fn, domds) in fn_d.items():
        print(dsname)
        m = get_means(varname, fn[grid], domds, freq, full_start_date, full_end_date, start_date, end_date, zoom_coords=zoom_coords, vardim=vardim)
        means_d[dsname] = m
        savez_data(m, f"{varname}_{dsname}_{freq}mean", start_date, end_date, restart_freq=restart_freq, fld=savez_fld, masked=masked)
    return means_d

def loadz_mean_fields (varname, loadz_fld, exp_list, freq, start_date, end_date, restart_freq='15d', masked=True, slices=(None,None,None,None)):
    data_d = loadz_data(f"{varname}*{freq}", loadz_fld, start_date, end_date, restart_freq=restart_freq, masked=masked, slices=slices)
    means_d = {}
    if type(exp_list) == list:
        for e in exp_list:
            means_d[e] = data_d[f"{varname}_{e}_{freq}mean"]
    else:
        means_d[exp_list] = data_d[f"{varname}_{freq}mean"]
    return means_d
    