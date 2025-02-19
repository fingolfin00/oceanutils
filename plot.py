import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as anim
# import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as tick
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import utils as ou

def plot_lonlat (ds, var, var_name, level, adjust_plt=False, vmin=None, vmax=None, zoom_idx=((None,None), (None,None)), zoom_coords=((None,None), (None,None)), cbar_ticks_num=10, cmap='jet', cmap_zerocentered='bwr', mask=None, point_idx=(None,None), point_coords=(None,None), point_clr='ko'):
    lat0_i, latf_i = ou.get_idx_from_lat(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lat(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]
    var_plt = var[level,lat0_i:latf_i,lon0_i:lonf_i]*mask[level,lat0_i:latf_i,lon0_i:lonf_i] if mask is not None else var[level,lat0_i:latf_i,lon0_i:lonf_i]
    PLT_LON, PLT_LAT = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], ou.create_latspace(ds)[lat0_i:latf_i])

    vmin_plt, vmax_plt = vmin if vmin else np.nanmin(var_plt[np.nonzero(var_plt)]), vmax if vmax else np.nanmax(var_plt[np.nonzero(var_plt)]) # zeros not counted
    if vmin_plt > 0 and vmax_plt > vmin_plt:
        vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    elif vmax_plt < 0 and vmin_plt < vmax_plt:
        vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    else:
        vcenter_plt = 0
        cmap = cmap_zerocentered
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(24,12)

    cs = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap, alpha=1)

    point_lat_i, point_lon_i = ou.get_idx_from_lat(point_coords[0], ds) if point_coords[0] else point_idx[0], ou.get_idx_from_lon(point_coords[1], ds) if point_coords[1] else point_idx[1]
    point_lat, point_lon = ou.get_lat_from_idx(point_idx[0], ds) if point_idx[0] else point_coords[0], ou.get_lon_from_idx(point_idx[1], ds) if point_idx[1] else point_coords[1]
    if point_lat_i and point_lon_i:
        ax.plot(PLT_LON[point_lon_i,point_lat_i], PLT_LAT[point_lon_i,point_lat_i], point_clr)
        ax.annotate(f"({point_lat:.2f}, {point_lon:.2f})", (PLT_LON[point_lon_i,point_lat_i], PLT_LAT[point_lon_i,point_lat_i]), xytext=(4,4), textcoords='offset points')

    cbar = fig.colorbar(cs, orientation="vertical", ticks=tick.LinearLocator(numticks=cbar_ticks_num))

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    zoom_str = f", [{lat0_i}:{latf_i},{lon0_i}:{lonf_i}]" if lat0_i or latf_i or lon0_i or lonf_i else ""
    ax.set_title(f"{var_name}, level {ou.create_levspace(ds)[level]:.3f}{zoom_str}")

    plt.show()

def plot_2d (ds, var, var_name, method='pcolor', contourf_levs=10, contour_step_level=100, contour_facecolor='grey', contour_lev_col=None, contour_zero_lev_label=False, adjust_plt=False, vmin=None, vmax=None, zoom_idx=((None,None), (None,None)), zoom_coords=((None,None), (None,None)), add_zoomstr_title=False, cbar=True, cbar_ticks_num=10, cbar_loc='right', cbar_title='', cmap='jet', cmap_zerocentered='bwr', mask=None, points_idx=[(None,None)], points_coords=[(None,None)], points_clr=['ko'], points_coords_ann=[None], points_idx_ann=[None], points_coords_ann_opts=[None], points_idx_ann_opts=[None]):
    lat0_i, latf_i = ou.get_idx_from_lat(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lat(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]
    var_plt = var[lat0_i:latf_i,lon0_i:lonf_i]*mask[lat0_i:latf_i,lon0_i:lonf_i] if mask is not None else var[lat0_i:latf_i,lon0_i:lonf_i]
    PLT_LON, PLT_LAT = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], ou.create_latspace(ds)[lat0_i:latf_i])

    vmin_plt, vmax_plt = vmin if vmin is not None else np.nanmin(var_plt[np.nonzero(var_plt)]), vmax if vmax is not None else np.nanmax(var_plt[np.nonzero(var_plt)]) # zeros not counted
    if vmin_plt >= 0 and vmax_plt > vmin_plt:
        vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    elif vmax_plt <= 0 and vmin_plt < vmax_plt:
        contour_neg_ls = 'solid'
        vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    else:
        vcenter_plt = 0
        cmap = cmap_zerocentered
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(24,12)

    match method:
        case 'pcolor':
            cs = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap, alpha=1)
        case 'contour' | 'filled_contour':
            def fmt(x):
                return f"{-x:.1f}"[:-2]
            if contour_lev_col:
                cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level), negative_linestyles=contour_neg_ls, colors=contour_lev_col, alpha=1)
            else:
                cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level), negative_linestyles=contour_neg_ls, cmap=cmap, alpha=1)
            if contour_zero_lev_label:
                cl = ax.clabel(cs, cs.levels, fmt=fmt, fontsize=10)
            else:
                cl = ax.clabel(cs, cs.levels[:-1], fmt=fmt, fontsize=10)
            # ax.set_bad(contour_facecolor)
            if method == 'filled_contour':
                cs = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level), cmap=cmap, alpha=1)
                ax.set_facecolor(contour_facecolor)
        case 'contourf':
            cs = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=contourf_levs, cmap=cmap, alpha=1)
            ax.set_facecolor(contour_facecolor)
        case _:
            print(f"Method {method} not supported")
            return

    for j, point_coords in enumerate(points_coords):
        if point_coords[0] and point_coords[1]:
            point_lat_i, point_lon_i = ou.get_idx_in_arr(point_coords[0], PLT_LAT[:,0]), ou.get_idx_in_arr(point_coords[1], PLT_LON[0,:])
            point_lat, point_lon = PLT_LAT[point_lat_i, 0], PLT_LON[0, point_lon_i]
            ax.plot(PLT_LON[point_lat_i,point_lon_i], PLT_LAT[point_lat_i,point_lon_i], points_clr[j])
            if points_coords_ann[j]:
                if points_coords_ann_opts[j]:
                    xytext = points_coords_ann_opts[j]['xytext']
                    bbox_d = points_coords_ann_opts[j]['bbox_d']
                    fontsize = points_coords_ann_opts[j]['fontsize']
                    weight = points_coords_ann_opts[j]['weight']
                    color = points_coords_ann_opts[j]['color']
                else:
                    xytext = (7,7)
                    bbox_d = dict(boxstyle="round", fc="w", alpha=0.4)
                    fontsize = 12
                    weight = 'bold'
                    color = 'black'
                ax.annotate(points_coords_ann[j], (PLT_LON[point_lat_i,point_lon_i], PLT_LAT[point_lat_i,point_lon_i]), xytext=xytext, textcoords='offset points', bbox=bbox_d, fontsize=fontsize, weight=weight, color=color)
            else:
                ax.annotate(f"({point_lat:.2f}, {point_lon:.2f})", (PLT_LON[point_lat_i,point_lon_i], PLT_LAT[point_lat_i,point_lon_i]), xytext=(4,4), textcoords='offset points')
    for j, point_idx in enumerate(points_idx):
        if point_idx[0] and point_idx[1]:
            point_lat_i, point_lon_i = point_idx[0], point_idx[1]
            point_lat, point_lon = PLT_LAT[point_lat_i, 0], PLT_LON[0, point_lon_i]
            ax.plot(PLT_LON[point_lat_i,point_lon_i], PLT_LAT[point_lat_i,point_lon_i], points_clr[j])
            if points_idx_ann[j]:
                if points_idx_ann_opts[j]:
                    xytext = points_idx_ann_opts[j]['xytext']
                    bbox_d = points_idx_ann_opts[j]['bbox_d']
                    fontsize = points_idx_ann_opts[j]['fontsize']
                    weight = points_idx_ann_opts[j]['weight']
                else:
                    xytext = (7,7)
                    bbox_d = dict(boxstyle="round", fc="w", alpha=0.4)
                    fontsize = 12
                    weight = 'bold'
                ax.annotate(points_coords_ann[j], (PLT_LON[point_lat_i,point_lon_i], PLT_LAT[point_lat_i,point_lon_i]), xytext=xytext, textcoords='offset points', bbox=bbox_d, fontsize=fontsize, weight=weight)
            else:
                ax.annotate(f"({point_lat:.2f}, {point_lon:.2f})", (PLT_LON[point_lat_i,point_lon_i], PLT_LAT[point_lat_i,point_lon_i]), xytext=(4,4), textcoords='offset points')

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_loc, size="2%", pad=0.05)
        cbar_ticks_num = contourf_levs+2 if method=='contourf' or method=='contour' else cbar_ticks_num
        print(cbar_ticks_num)
        cb = fig.colorbar(cs, cax=cax, ticks=tick.LinearLocator(numticks=cbar_ticks_num))
        cb.ax.set_title(cbar_title, fontsize=8)

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    if add_zoomstr_title:
        zoom_str = f", [{lat0_i}:{latf_i},{lon0_i}:{lonf_i}]" if lat0_i or latf_i or lon0_i or lonf_i else ""
    else:
        zoom_str = ""
    ax.set_title(f"{var_name}{zoom_str}")

    plt.show()

def plot_lonlev (ds, var, var_name, lat_i, lev_factor=1, adjust_plt=False, vmin=None, vmax=None, zoom_idx=((None,None), (None,None)), zoom_coords=((None,None), (None,None)), cbar_ticks_num=10, cmap='jet', cmap_zerocentered='bwr', mask=None, point_idx=(None,None), point_coords=(None,None), point_clr='ko'):
    lev0_i, levf_i = ou.get_idx_from_lev(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lev(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]
    var_plt = var[lev0_i:levf_i,lat_i,lon0_i:lonf_i]*mask[lev0_i:levf_i,lat_i,lon0_i:lonf_i] if mask is not None else var[lev0_i:levf_i,lat_i,lon0_i:lonf_i]
    PLT_LON, PLT_LEV = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], lev_factor*ou.create_levspace(ds)[lev0_i:levf_i])

    vmin_plt, vmax_plt = vmin if vmin else np.nanmin(var_plt[np.nonzero(var_plt)]), vmax if vmax else np.nanmax(var_plt[np.nonzero(var_plt)]) # zeros not counted
    if vmin_plt > 0 and vmax_plt > vmin_plt:
        vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    elif vmax_plt < 0 and vmin_plt < vmax_plt:
        vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    else:
        vcenter_plt = 0
        cmap = cmap_zerocentered
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(24,12)

    cs = ax.pcolormesh(PLT_LON, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap, alpha=1)

    point_lon_i, point_lev_i = ou.get_idx_from_lon(point_coords[0], ds) if point_coords[0] else point_idx[0], ou.get_idx_from_lev(point_coords[1], ds) if point_coords[1] else point_idx[1]
    point_lon, point_lev = ou.get_lon_from_idx(point_idx[1], ds) if point_idx[1] else point_coords[1], ou.get_lev_from_idx(point_idx[0], ds) if point_idx[0] else point_coords[0]
    # print(point_lon, point_lev)
    if point_lev and point_lon:
        ax.plot(PLT_LON[point_lon_i,point_lev_i], PLT_LEV[point_lon_i,point_lev_i], point_clr)
        ax.annotate(f"({point_lon:.2f}, {point_lev:.2f})", (PLT_LON[point_lon_i,point_lev_i], PLT_LEV[point_lon_i,point_lev_i]), xytext=(4,4), textcoords='offset points')

    cbar = fig.colorbar(cs, orientation="vertical", ticks=tick.LinearLocator(numticks=cbar_ticks_num))

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("lon")
    ax.set_ylabel("lev")

    zoom_str = f", [{lon0_i}:{lonf_i},{lev0_i}:{levf_i}]" if lev0_i or levf_i or lon0_i or lonf_i else ""
    ax.set_title(f"{var_name}, lat {ou.create_latspace(ds)[lat_i]:.3f}{zoom_str}")

    plt.show()