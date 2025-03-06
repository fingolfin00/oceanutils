import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.animation as anim
# import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm, LogNorm, BoundaryNorm
import matplotlib.ticker as tick
import matplotlib.dates as pldates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import utils as ou

def get_cbar_ticks_num (vmax, vmin, contour_step_level):
    return int(np.abs(vmax-vmin)//contour_step_level)+2

def plot_lonlat (ds, var, var_name, level, keep_var_dim=False, savefig_fn=None, savefig_resfac=1, hres=24, vres=12, add_lev_str=False, anomaly=False,
                 adjust_plt=False, vmin=None, vmax=None, fmt=None, method='pcolor', contour_lev_col=None, contour_step_level=100,
                 contour_zero_lev_label=False,
                 zoom_idx=((None,None), (None,None)), zoom_coords=((None,None), (None,None)), add_zoomstr_title=False, noiplot=False,
                 cbar=True, cbar_ticks_num=10, cbar_loc='right', cbar_title='', contour_facecolor='grey', quiver=False, extend='neither',
                 cmap='jet', mask=None, point_idx=(None,None), point_coords=(None,None), point_clr='ko'):
    
    lat0_i, latf_i = ou.get_idx_from_lat(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lat(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]
    print(lon0_i, lonf_i)
    if keep_var_dim:
        var_plt = var[level,:,:]*mask[level,:,:] if mask is not None else var[level,:,:]
    else:
        var_plt = var[level,lat0_i:latf_i,lon0_i:lonf_i]*mask[level,lat0_i:latf_i,lon0_i:lonf_i] if mask is not None else var[level,lat0_i:latf_i,lon0_i:lonf_i]

    if anomaly:
        var_plt = var_plt-np.ma.mean(var_plt)
    
    PLT_LON, PLT_LAT = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], ou.create_latspace(ds)[lat0_i:latf_i])

    vmin_plt, vmax_plt = np.ma.min(var_plt) if vmin is None else vmin, np.ma.max(var_plt) if vmax is None else vmax  # zeros not counted
    vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(hres,vres)
    if noiplot:
        plt.ioff()

    if type(contour_step_level) == list:
        levels = contour_step_level
    else:
        levels = np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level)

    contour_neg_ls = 'solid'
    if method == 'pcolor':
            pc = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap, alpha=1)
    elif method == 'contour' or method == 'filled_contour' or method == 'contour_pcolor':
        if fmt is None:
            def fmt(x):
                return f"{-x:.1f}"[:-2]
        if contour_lev_col:
            cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels, extend=extend, negative_linestyles=contour_neg_ls, colors=contour_lev_col, alpha=1)
        else:
            cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels, extend=extend, negative_linestyles=contour_neg_ls, cmap=cmap, alpha=1)
        if contour_zero_lev_label:
            cl = ax.clabel(cs, cs.levels, fmt=fmt, fontsize=10)
        else:
            cl = ax.clabel(cs, cs.levels[:-1], fmt=fmt, fontsize=10)
        # ax.set_bad(contour_facecolor)
        if method == 'filled_contour':
            csf = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels, extend=extend, cmap=cmap, alpha=1)
    elif method == 'contourf':
        csf = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels, extend=extend, cmap=cmap, alpha=1)
    else:
        print(f"Method {method} not supported")
        return
    ax.set_facecolor(contour_facecolor)
    # cs = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap, alpha=1)

    point_lat_i, point_lon_i = ou.get_idx_from_lat(point_coords[0], ds) if point_coords[0] else point_idx[0], ou.get_idx_from_lon(point_coords[1], ds) if point_coords[1] else point_idx[1]
    point_lat, point_lon = ou.get_lat_from_idx(point_idx[0], ds) if point_idx[0] else point_coords[0], ou.get_lon_from_idx(point_idx[1], ds) if point_idx[1] else point_coords[1]
    if point_lat_i and point_lon_i:
        ax.plot(PLT_LON[point_lon_i,point_lat_i], PLT_LAT[point_lon_i,point_lat_i], point_clr)
        ax.annotate(f"({point_lat:.2f}, {point_lon:.2f})", (PLT_LON[point_lon_i,point_lat_i], PLT_LAT[point_lon_i,point_lat_i]), xytext=(4,4), textcoords='offset points')

    if quiver:
        u_quiver = quiver['u'][lat0_i:latf_i,lon0_i:lonf_i]
        v_quiver = quiver['v'][lat0_i:latf_i,lon0_i:lonf_i]
        s = quiver['strides']
        Q = ax.quiver(PLT_LON[::s,::s], PLT_LAT[::s,::s], u_quiver[::s,::s], v_quiver[::s,::s], units=quiver['units'], width=quiver['width'], color=quiver['color'],
                     headwidth=quiver['headwidth'], scale=quiver['scale'], headlength=quiver['headlength'], headaxislength=quiver['headaxislength'], visible=True)

    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_loc, size="2%", pad=0.05)
        # cbar_ticks_num = contourf_levs+2 if method=='contourf' or method=='contour' or method=='filled_contour' else cbar_ticks_num
        # print(cbar_ticks_num)
        cb = fig.colorbar(cs, cax=cax, ticks=tick.LinearLocator(numticks=cbar_ticks_num))
        cb.ax.set_title(cbar_title, fontsize=8)

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    if add_zoomstr_title:
        zoom_str = f", [{lat0_i}:{latf_i},{lon0_i}:{lonf_i}]" if lat0_i or latf_i or lon0_i or lonf_i else ""
    else:
        zoom_str = ""
    lev_str = f", level {ou.create_levspace(ds)[level]:.3f}" if add_lev_str else ""
    ax.set_title(f"{var_name}{lev_str}{zoom_str}")

    if savefig_fn:
        plt.savefig(savefig_fn, dpi=savefig_resfac*hres*vres, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return (fig, ax)


def plot_2d (ds, var, var_name, savefig_fn=None, savefig_resfac=1, hres=24, vres=12, method='pcolor', keep_var_dim=False, anomaly=False,
             contour_step_level=100, contour_facecolor='grey', contour_lev_col=None, contour_zero_lev_label=False, contour_color_factor=None,
             adjust_plt=False, fmt=None, vmin=None, vmax=None, strides=None, contour_lev_labels=True, contour_linewidths=1.5, contour_linestyles=None,
             zoom_idx=((None,None), (None,None)), zoom_coords=((None,None), (None,None)), add_zoomstr_title=False, extend='neither',
             cbar=True, cbar_ticks_num=10, cbar_loc='right', cbar_title='', cmap='jet', mask=None, noiplot=False,
             points_idx=[(None,None)], points_coords=[(None,None)], points_clr=['ko'], points_coords_ann=[None], points_idx_ann=[None], points_coords_ann_opts=[None], points_idx_ann_opts=[None]):
    
    lat0_i, latf_i = ou.get_idx_from_lat(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lat(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]

    if keep_var_dim:
        var_plt = var*mask if mask is not None else var
    else:
        var_plt = var[lat0_i:latf_i,lon0_i:lonf_i]*mask[lat0_i:latf_i,lon0_i:lonf_i] if mask is not None else var[lat0_i:latf_i,lon0_i:lonf_i]

    if anomaly:
        var_plt = var_plt-np.ma.mean(var_plt)
    
    PLT_LON, PLT_LAT = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], ou.create_latspace(ds)[lat0_i:latf_i])

    if strides:
        var_plt = var_plt[::strides,::strides]
        PLT_LON, PLT_LAT = PLT_LON[::strides,::strides], PLT_LAT[::strides,::strides]

    vmin_plt, vmax_plt = np.ma.min(var_plt) if vmin is None else vmin, np.ma.max(var_plt) if vmax is None else vmax
    vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(hres,vres)
    if noiplot:
        plt.ioff()
    
    if type(contour_step_level) == list:
        levels = contour_step_level
        ncolors = len(contour_step_level)*contour_color_factor if contour_color_factor else len(contour_step_level)
        norm = BoundaryNorm(contour_step_level, ncolors)
    else:
        levels = np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level)
        norm = TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt)

    contour_neg_ls = 'solid'
    if method == 'pcolor':
        pc = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=norm, cmap=cmap, alpha=1)
    elif method == 'contour' or method == 'filled_contour' or method == 'contour_pcolor':
        if fmt is None:
            def fmt(x):
                return f"{-x:.1f}"[:-2]
        if contour_lev_col:
            cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, negative_linestyles=contour_neg_ls, colors=contour_lev_col, alpha=1, extend=extend, linewidths=contour_linewidths, linestyles=contour_linestyles)
        else:
            cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, negative_linestyles=contour_neg_ls, cmap=cmap, alpha=1, extend=extend, linewidths=contour_linewidths, linestyles=contour_linestyles)
        if contour_lev_labels:
            if contour_zero_lev_label:
                cl = ax.clabel(cs, cs.levels, fmt=fmt, fontsize=10)
            else:
                cl = ax.clabel(cs, cs.levels[:-1], fmt=fmt, fontsize=10)
        # ax.set_bad(contour_facecolor)
        if method == 'filled_contour':
            csf = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, cmap=cmap, alpha=1, extend=extend)
        if method == 'contour_pcolor':
            pc = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=norm, cmap=cmap, alpha=1)
    elif method == 'contourf':
        csf = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, cmap=cmap, alpha=1, extend=extend)
    else:
        print(f"Method {method} not supported")
        return
    ax.set_facecolor(contour_facecolor)
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
        if method == 'contourf' or method == 'filled_contour':
            e = csf
        elif method == 'pcolor' or method == 'contour_pcolor':
            e = pc
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_loc, size="2%", pad=0.05)
        # cbar_ticks_num = contourf_levs+2 if method=='contourf' or method=='contour' or method=='filled_contour' else cbar_ticks_num
        # print(cbar_ticks_num)
        ticks = tick.FixedLocator(contour_step_level) if type(contour_step_level)==list else tick.LinearLocator(numticks=cbar_ticks_num)
        cb = fig.colorbar(e, cax=cax, ticks=ticks)
        cb.ax.set_title(cbar_title, fontsize=8)

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")

    if add_zoomstr_title:
        zoom_str = f", [{lat0_i}:{latf_i},{lon0_i}:{lonf_i}]" if lat0_i or latf_i or lon0_i or lonf_i else ""
    else:
        zoom_str = ""
    ax.set_title(f"{var_name}{zoom_str}")

    if savefig_fn:
        plt.savefig(savefig_fn, dpi=savefig_resfac*hres*vres, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return (fig, ax)


def plot_lonlev (ds, var, var_name, lat_i, keep_var_dim=False, savefig_fn=None, savefig_resfac=1, hres=24, vres=12, anomaly=False,
                 lev_factor=1, adjust_plt=False, add_lat_str=False, vmin=None, vmax=None, extend='neither', method='pcolor',
                 contour_step_level=100, contour_facecolor='grey', contour_lev_col=None, contour_zero_lev_label=False, fmt=None,
                 zoom_idx=((None,None), (None,None)), zoom_coords=((None,None), (None,None)), add_zoomstr_title=False, cbar=True, cbar_loc='right',
                 cbar_ticks_num=10, cbar_title='', cmap='jet', mask=None, point_idx=(None,None), point_coords=(None,None), point_clr='ko', noiplot=False):
    
    lev0_i, levf_i = ou.get_idx_from_lev(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lev(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]
    print(lev0_i, levf_i)
    print(lon0_i, lonf_i)

    if keep_var_dim:
        var_plt = var[:,lat_i,:]*mask[:,lat_i,:] if mask is not None else var[:,lat_i,:]
    else:
        var_plt = var[lev0_i:levf_i,lat_i,lon0_i:lonf_i]*mask[lev0_i:levf_i,lat_i,lon0_i:lonf_i] if mask is not None else var[lev0_i:levf_i,lat_i,lon0_i:lonf_i]

    if anomaly:
        var_plt = var_plt-np.ma.mean(var_plt)

    vmin_plt, vmax_plt = np.ma.min(var_plt) if vmin is None else vmin, np.ma.max(var_plt) if vmax is None else vmax
    vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    PLT_LON, PLT_LEV = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], lev_factor*ou.create_levspace(ds)[lev0_i:levf_i])
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")

    if noiplot:
        plt.ioff()
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(hres,vres)

    if type(contour_step_level) == list:
        levels = contour_step_level
        ncolors = len(contour_step_level)*contour_color_factor if contour_color_factor else len(contour_step_level)
        norm = BoundaryNorm(contour_step_level, ncolors)
    else:
        levels = np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level)
        norm = TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt)

    contour_neg_ls = 'solid'
    if method == 'pcolor':
            pc = ax.pcolormesh(PLT_LON, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap, alpha=1)
    elif method == 'contour' or method == 'filled_contour' or method == 'contour_pcolor':
        if fmt is None:
            def fmt(x):
                return f"{-x:.1f}"[:-2]
        if contour_lev_col:
            cs = ax.contour(PLT_LON, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels,
                            extend=extend, negative_linestyles=contour_neg_ls, colors=contour_lev_col, alpha=1)
        else:
            cs = ax.contour(PLT_LON, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels,
                            extend=extend, negative_linestyles=contour_neg_ls, cmap=cmap, alpha=1)
        if contour_zero_lev_label:
            cl = ax.clabel(cs, cs.levels, fmt=fmt, fontsize=10)
        else:
            cl = ax.clabel(cs, cs.levels[:-1], fmt=fmt, fontsize=10)
        # ax.set_bad(contour_facecolor)
        if method == 'filled_contour':
            csf = ax.contourf(PLT_LON, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt),
                             extend=extend, levels=levels, cmap=cmap, alpha=1)
        if method == 'contour_pcolor':
            pc = ax.pcolormesh(PLT_LON, PLT_LEV, var_plt, norm=norm, cmap=cmap, alpha=1)
    elif method == 'contourf':
        csf = ax.contourf(PLT_LON, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt),
                         extend=extend, levels=levels, cmap=cmap, alpha=1)
    else:
        print(f"Method {method} not supported")
        return
    ax.set_facecolor(contour_facecolor)
    # ax.set_yscale('log')
    point_lon_i, point_lev_i = ou.get_idx_from_lon(point_coords[0], ds) if point_coords[0] else point_idx[0], ou.get_idx_from_lev(point_coords[1], ds) if point_coords[1] else point_idx[1]
    point_lon, point_lev = ou.get_lon_from_idx(point_idx[1], ds) if point_idx[1] else point_coords[1], ou.get_lev_from_idx(point_idx[0], ds) if point_idx[0] else point_coords[0]
    # print(point_lon, point_lev)
    if point_lev and point_lon:
        ax.plot(PLT_LON[point_lon_i,point_lev_i], PLT_LEV[point_lon_i,point_lev_i], point_clr)
        ax.annotate(f"({point_lon:.2f}, {point_lev:.2f})", (PLT_LON[point_lon_i,point_lev_i], PLT_LEV[point_lon_i,point_lev_i]), xytext=(4,4), textcoords='offset points')

    if cbar:
        if method == 'contourf' or method == 'filled_contour':
            e = csf
        elif method == 'pcolor' or method == 'contour_pcolor':
            e = pc
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_loc, size="2%", pad=0.05)
        # cbar_ticks_num = contourf_levs+2 if method=='contourf' or method=='contour' or method=='filled_contour' else cbar_ticks_num
        # print(cbar_ticks_num)
        cb = fig.colorbar(e, cax=cax, ticks=tick.LinearLocator(numticks=cbar_ticks_num))
        cb.ax.set_title(cbar_title, fontsize=8)
    # ax.set_facecolor('gray')

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("lon")
    ax.set_ylabel("lev")

    if add_zoomstr_title:
        zoom_str = f", [{lon0_i}:{lonf_i},{lev0_i}:{levf_i}]" if lev0_i or levf_i or lon0_i or lonf_i else ""
    else:
        zoom_str = ""
    lat_str = f", lat {ou.create_latspace(ds)[lat_i]:.3f}" if add_lat_str else ""
    ax.set_title(f"{var_name}{lat_str}{zoom_str}")

    if savefig_fn:
        plt.savefig(savefig_fn, dpi=savefig_resfac*hres*vres, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    if method == 'contourf' or method == 'filled_contour':
        e = csf
    elif method == 'pcolor' or method == 'contour_pcolor':
        e = pc
    else:
        e = cs
    return e


def plot_hoevmoller (ds, var, start_date, end_date, var_name, restart_freq='6h', keep_var_dim=False, method='pcolor', fmt=None,
                     contour_step_level=100, contour_facecolor='grey', contour_lev_col=None, contour_zero_lev_label=False,
                     savefig_fn=None, savefig_resfac=1, hres=24, vres=12, extend='neither',
                     lev_factor=1, adjust_plt=False, vmin=None, vmax=None,
                     zoom_idx=((None,None),), zoom_coords=((None,None),), add_zoomstr_title=False, cbar=True, cbar_loc='right',
                     cbar_ticks_num=10, cbar_title='', cmap='jet', mask=None, point_idx=(None,None), point_coords=(None,None), point_clr='ko'):
    '''Generate a Hoevmoller pcolormesh plot of a 2d variable (time x level) with time in the x axis and level depth in the y axis'''
    
    lev0_i, levf_i = ou.get_idx_from_lev(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lev(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    # lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]
    # print(lev0_i, levf_i)
    restart_dates = pd.date_range(start=start_date, end=end_date, freq=restart_freq).to_pydatetime().tolist()[:-1]
    # restart_dates = np.arange(np.shape(var)[0])
    start_date, end_date = ou.from_date_to_datetime(start_date), ou.from_date_to_datetime(end_date)
    
    if keep_var_dim:
        var_plt = np.ma.swapaxes(var*mask,0,1) if mask is not None else np.ma.swapaxes(var,0,1)
    else:
        var_plt = np.ma.swapaxes(var[:,lev0_i:levf_i]*mask[:,lev0_i:levf_i],0,1) if mask is not None else np.ma.swapaxes(var[:,lev0_i:levf_i],0,1)
    PLT_TIM, PLT_LEV = np.meshgrid(restart_dates, lev_factor*ou.create_levspace(ds)[lev0_i:levf_i])

    vmin_plt, vmax_plt = np.ma.min(var_plt) if vmin is None else vmin, np.ma.max(var_plt) if vmax is None else vmax  # zeros not counted
    vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(hres,vres)

    if type(contour_step_level) == list:
        levels = contour_step_level
    else:
        levels = np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level)

    contour_neg_ls = 'solid'
    if method == 'pcolor':
            pc = ax.pcolormesh(PLT_TIM, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), cmap=cmap, alpha=1)
    elif method == 'contour' or method == 'filled_contour' or method == 'contour_pcolor':
        if fmt is None:
            def fmt(x):
                return f"{-x:.1f}"[:-2]
        if contour_lev_col:
            cs = ax.contour(PLT_TIM, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels,
                            extend=extend, negative_linestyles=contour_neg_ls, colors=contour_lev_col, alpha=1)
        else:
            cs = ax.contour(PLT_TIM, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt), levels=levels,
                            extend=extend, negative_linestyles=contour_neg_ls, cmap=cmap, alpha=1)
        if contour_zero_lev_label:
            cl = ax.clabel(cs, cs.levels, fmt=fmt, fontsize=10)
        else:
            cl = ax.clabel(cs, cs.levels[:-1], fmt=fmt, fontsize=10)
        # ax.set_bad(contour_facecolor)
        if method == 'filled_contour':
            csf = ax.contourf(PLT_TIM, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt),
                             extend=extend, levels=levels, cmap=cmap, alpha=1)
        if method == 'contour_pcolor':
            pc = ax.pcolormesh(PLT_TIM, PLT_LEV, var_plt, norm=norm, cmap=cmap, alpha=1)
    elif method == 'contourf':
        csf = ax.contourf(PLT_TIM, PLT_LEV, var_plt, norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt),
                         extend=extend, levels=levels, cmap=cmap, alpha=1)
    else:
        print(f"Method {method} not supported")
        return
    ax.set_facecolor(contour_facecolor)
    # cs = ax.pcolormesh(PLT_TIM, PLT_LEV, np.swapaxes(var_plt,0,1),
    #                    norm=TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt),
    #                    # norm=LogNorm(vmin=vmin_plt, vmax=vmax_plt),
    #                    cmap=cmap, alpha=1)
    # ax.set_yscale('log')
    point_time_i, point_lev_i = ou.get_idx_in_arr(point_coords[0], restart_dates) if point_coords[0] else point_idx[0], ou.get_idx_from_lev(point_coords[1], ds) if point_coords[1] else point_idx[1]
    point_time, point_lev = ou.get_lon_from_idx(point_idx[1], ds) if point_idx[1] else point_coords[1], ou.get_lev_from_idx(point_idx[0], ds) if point_idx[0] else point_coords[0]
    # print(point_time, point_lev)
    if point_lev and point_time:
        ax.plot(PLT_TIM[point_time_i,point_lev_i], PLT_LEV[point_time_i,point_lev_i], point_clr)
        ax.annotate(f"({point_time:.2f}, {point_lev:.2f})", (PLT_LON[point_time_i,point_lev_i], PLT_LEV[point_time_i,point_lev_i]), xytext=(4,4), textcoords='offset points')

    if cbar:
        if method == 'contourf' or method == 'filled_contour':
            e = csf
        elif method == 'pcolor' or method == 'contour_pcolor':
            e = pc
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(cbar_loc, size="2%", pad=0.05)
        # cbar_ticks_num = contourf_levs+2 if method=='contourf' or method=='contour' or method=='filled_contour' else cbar_ticks_num
        # print(cbar_ticks_num)
        cb = fig.colorbar(cs, cax=cax, ticks=tick.LinearLocator(numticks=cbar_ticks_num))
        cb.ax.set_title(cbar_title, fontsize=8)
    # ax.set_facecolor('gray')

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    # ax.set_xlabel("time")
    ax.set_ylabel("level [m]")

    if add_zoomstr_title:
        zoom_str = f", [{lev0_i}:{levf_i}]" if lev0_i or levf_i else ""
    else:
        zoom_str = ""
    
    ax.set_title(f"{var_name}{zoom_str}")

    ax.xaxis.set_major_formatter(pldates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(pldates.MonthLocator())

    if savefig_fn:
        plt.savefig(savefig_fn, dpi=savefig_resfac*hres*vres, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return (fig, ax)


def save_lonlev_pictures (ds, var, savefig_varname, start_date, end_date, lat, method='pcolor', vmin=None, vmax=None, mask=None,
                          cmap='jet', adjust_plt=False, hres=24, vres=12,
                          zoom_coords=((None,None), (None,None)), save_fld='./', savefreq='m', noiplot=True):

    os.makedirs(save_fld, exist_ok=True)
    if savefreq == 'm':
        savefig_range = pd.date_range(start=start_date, end=end_date, freq='1M').to_pydatetime().tolist()
        savefig_restart_fmt = ou.get_restart_strformat('m')
    elif savefreq == 'd':
        savefig_range = pd.date_range(start=start_date, end=end_date, freq='1d').to_pydatetime().tolist()
        savefig_restart_fmt = ou.get_restart_strformat('d')
    lat_idx = ou.get_idx_from_lat(lat, ds)
    lev0, levf = zoom_coords[0][0], zoom_coords[0][1]
    lon0, lonf = zoom_coords[1][0], zoom_coords[1][1]
    for d in savefig_range:
        i = savefig_range.index(d)
        # print(i)
        var_plt = var[i,:,:,:]
        savefig_period = d.strftime(savefig_restart_fmt)
        plot_lonlev(ds, var_plt, '', lat_idx, cmap=cmap, cmap_zerocentered=cmap_zerocentered, adjust_plt=adjust_plt, hres=hres, vres=vres,
                         # add_lat_str=True,
                         savefig_fn=f"{save_fld}{savefig_varname}-{ou.get_lat_from_idx(lat_idx, ds)}_{savefig_period}.png",
                         vmin=vmin, vmax=vmax, mask=None, lev_factor=-1, zoom_coords=((lev0,levf),(lon0, lonf)), noiplot=noiplot)

def save_lonlat_pictures (ds, var, savefig_varname, start_date, end_date, level, method='pcolor', fmt=None, extend='neither',
                          cbar_ticks_num=10, contour_step_level=100, contour_facecolor='grey', contour_lev_col=None, contour_zero_lev_label=False,
                          cmap='jet', adjust_plt=False, anomaly=False, hres=24, vres=12,
                          keep_var_dim=False, vmin=None, vmax=None, mask=None, zoom_coords=((None,None), (None,None)),
                          save_fld='./', savefreq='m', vardim='3d', noiplot=True):
    
    os.makedirs(save_fld, exist_ok=True)
    if savefreq == 'm':
        savefig_range = pd.date_range(start=start_date, end=end_date, freq='1M').to_pydatetime().tolist()
        savefig_restart_fmt = ou.get_restart_strformat('m')
    elif savefreq == 'd':
        savefig_range = pd.date_range(start=start_date, end=end_date, freq='1d').to_pydatetime().tolist()
        savefig_restart_fmt = ou.get_restart_strformat('d')
    lev_idx = ou.get_idx_from_lev(level, ds)
    lat0, latf = zoom_coords[0][0], zoom_coords[0][1]
    lon0, lonf = zoom_coords[1][0], zoom_coords[1][1]
    for d in savefig_range:
        i = savefig_range.index(d)
        # print(i)
        var_plt = var[i,:,:,:] if vardim == '3d' else var[i,:,:]
        savefig_period = d.strftime(savefig_restart_fmt)
        if vardim == '3d':
            plot_lonlat(ds, var_plt, '', lev_idx, keep_var_dim=keep_var_dim, cmap=cmap, adjust_plt=adjust_plt, anomaly=anomaly, extend=extend,
                        method=method, contour_step_level=contour_step_level, contour_facecolor=contour_facecolor, hres=hres, vres=vres,
                        contour_lev_col=contour_lev_col, contour_zero_lev_label=contour_zero_lev_label, fmt=fmt, cbar_ticks_num=cbar_ticks_num,
                        savefig_fn=f"{save_fld}{savefig_varname}-{ou.get_lev_from_idx(lev_idx, ds)}_{savefig_period}.png",
                        vmin=vmin, vmax=vmax, mask=None, zoom_coords=((lat0,latf),(lon0, lonf)), noiplot=noiplot)
        else:
            plot_2d(ds, var_plt, '', keep_var_dim=keep_var_dim, cmap=cmap, adjust_plt=adjust_plt, anomaly=anomaly, extend=extend,
                    method=method, contour_step_level=contour_step_level, contour_facecolor=contour_facecolor, hres=hres, vres=vres,
                    contour_lev_col=contour_lev_col, contour_zero_lev_label=contour_zero_lev_label, fmt=fmt, cbar_ticks_num=cbar_ticks_num,
                    savefig_fn=f"{save_fld}{savefig_varname}_{savefig_period}.png",
                    vmin=vmin, vmax=vmax, mask=None, zoom_coords=((lat0,latf),(lon0, lonf)), noiplot=noiplot)

def plot_bathy (ds, ax, lati, latf, loni, lonf, vmin, vcenter, vmax,
                elev, azim, roll, bathy_var=None, cmap='Blues'): 
    lat0_i, latf_i = ou.get_idx_from_lat(lati, ds), ou.get_idx_from_lat(latf, ds)
    lon0_i, lonf_i = ou.get_idx_from_lon(loni, ds), ou.get_idx_from_lon(lonf, ds)
    PLT_LON, PLT_LAT = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], ou.create_latspace(ds)[lat0_i:latf_i])
    if bathy_var is None:
        bathy_var = ou.create_levspace(ds)
    if len(np.shape(bathy_var)) == 2:
        X, Y, Z = PLT_LON, PLT_LAT, bathy_var[lat0_i:latf_i,lon0_i:lonf_i]
    else:
        # print(np.sum(np.shape(PLT_LON)))
        # levspaceflat = [levspace]*(np.prod(np.shape(PLT_LON)))
        # levspace2d = np.array(levspaceflat).reshape(np.shape(levspace)+np.shape(PLT_LON))
        X, Y, Z = PLT_LON, PLT_LAT, bathy_var[lat0_i:latf_i,lon0_i:lonf_i]
    
    
    # colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
    # my_cmap = ListedColormap(colors, name="my_cmap")
    ax.plot_surface(X, Y, Z,
                    # vmin=vmin_plt, vmax=vmax_plt,
                    norm=TwoSlopeNorm(vmin=vmin, vmax=vmax, vcenter=vcenter),
                    cmap=cmap,
                    # cmap = 'Blues',
                    # color='blue',
                    cstride=1, rstride=1, alpha=.9, antialiased=True)
    
    ax.set_proj_type('persp')  # FOV = 0 deg
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    return ax

def create_cbar (fig, ax, cs, cbar_ticks_num=10, cbar_loc='right', cbar_title=''):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(cbar_loc, size="2%", pad=0.05)
    # cbar_ticks_num = contourf_levs+2 if method=='contourf' or method=='contour' or method=='filled_contour' else cbar_ticks_num
    # print(cbar_ticks_num)
    ticks = tick.FixedLocator(contour_step_level) if type(contour_step_level)==list else tick.LinearLocator(numticks=cbar_ticks_num)
    cb = fig.colorbar(cs, cax=cax, ticks=ticks)
    cb.ax.set_title(cbar_title, fontsize=8)

def plot_2d_ax (
    ds, var, ax, method='pcolor', keep_var_dim=False, anomaly=False, strides=None, mask=None, vmin=None, vmax=None,
    adjust_plt=False, extend='neither', cmap='jet', quiver=False, stream=False,
    zoom_idx=((None,None), (None,None)), zoom_coords=((None,None), (None,None)),
    contour_step_level=None, contour_facecolor='grey', contour_lev_col=None, contour_zero_lev_label=False, contour_color_factor=None,
    contour_label_fmt=None, contour_lev_labels=True, contour_linewidths=1.5, contour_linestyles=None,
    points_idx=[(None,None)], points_coords=[(None,None)], points_clr=['ko'], points_coords_ann=[None], points_idx_ann=[None], points_coords_ann_opts=[None], points_idx_ann_opts=[None]
):
    
    lat0_i, latf_i = ou.get_idx_from_lat(zoom_coords[0][0], ds) if zoom_coords[0][0] else zoom_idx[0][0], ou.get_idx_from_lat(zoom_coords[0][1], ds) if zoom_coords[0][1] else zoom_idx[0][1]
    lon0_i, lonf_i = ou.get_idx_from_lon(zoom_coords[1][0], ds) if zoom_coords[1][0] else zoom_idx[1][0], ou.get_idx_from_lon(zoom_coords[1][1], ds) if zoom_coords[1][1] else zoom_idx[1][1]

    if keep_var_dim:
        var_plt = var*mask if mask is not None else var
    else:
        var_plt = var[lat0_i:latf_i,lon0_i:lonf_i]*mask[lat0_i:latf_i,lon0_i:lonf_i] if mask is not None else var[lat0_i:latf_i,lon0_i:lonf_i]

    if anomaly:
        var_plt = var_plt-np.ma.mean(var_plt)
    
    PLT_LON, PLT_LAT = np.meshgrid(ou.create_lonspace(ds)[lon0_i:lonf_i], ou.create_latspace(ds)[lat0_i:latf_i])

    if strides:
        var_plt = var_plt[::strides,::strides]
        PLT_LON, PLT_LAT = PLT_LON[::strides,::strides], PLT_LAT[::strides,::strides]

    vmin_plt, vmax_plt = np.ma.min(var_plt) if vmin is None else vmin, np.ma.max(var_plt) if vmax is None else vmax
    vcenter_plt = vmin_plt+(vmax_plt-vmin_plt)/2
    print(f"vmin, vcenter, vmax = ({vmin_plt:.3f}, {vcenter_plt:.3f}, {vmax_plt:.3f})")
    
    if type(contour_step_level) == list:
        levels = contour_step_level
        ncolors = len(contour_step_level)*contour_color_factor if contour_color_factor else len(contour_step_level)
        norm = BoundaryNorm(contour_step_level, ncolors)
    elif type(contour_step_level) == int:
        levels = np.arange(vmin_plt,vmax_plt+contour_step_level,contour_step_level)
        norm = TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt)
    else:
        levels = None
        norm = TwoSlopeNorm(vmin=vmin_plt, vmax=vmax_plt, vcenter=vcenter_plt)

    contour_neg_ls = 'solid'
    if method == 'pcolor':
        pc = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=norm, cmap=cmap, alpha=1)
    elif method == 'contour' or method == 'filled_contour' or method == 'contour_pcolor':
        if contour_label_fmt is None:
            def contour_label_fmt(x):
                return f"{-x:.1f}"[:-2]
        if contour_lev_col:
            cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, negative_linestyles=contour_neg_ls, colors=contour_lev_col, alpha=1, extend=extend, linewidths=contour_linewidths, linestyles=contour_linestyles)
        else:
            cs = ax.contour(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, negative_linestyles=contour_neg_ls, cmap=cmap, alpha=1, extend=extend, linewidths=contour_linewidths, linestyles=contour_linestyles)
        if contour_lev_labels:
            if contour_zero_lev_label:
                cl = ax.clabel(cs, cs.levels, fmt=contour_label_fmt, fontsize=10)
            else:
                cl = ax.clabel(cs, cs.levels[:-1], fmt=contour_label_fmt, fontsize=10)
        # ax.set_bad(contour_facecolor)
        if method == 'filled_contour':
            csf = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, cmap=cmap, alpha=1, extend=extend)
        if method == 'contour_pcolor':
            pc = ax.pcolormesh(PLT_LON, PLT_LAT, var_plt, norm=norm, cmap=cmap, alpha=1)
    elif method == 'contourf':
        csf = ax.contourf(PLT_LON, PLT_LAT, var_plt, norm=norm, levels=levels, cmap=cmap, alpha=1, extend=extend)
    else:
        print(f"Method {method} not supported")
        return
    ax.set_facecolor(contour_facecolor)
    if quiver:
        u_quiver = quiver['u'][lat0_i:latf_i,lon0_i:lonf_i]
        v_quiver = quiver['v'][lat0_i:latf_i,lon0_i:lonf_i]
        s = quiver['strides']
        Q = ax.quiver(PLT_LON[::s,::s], PLT_LAT[::s,::s], u_quiver[::s,::s], v_quiver[::s,::s], units=quiver['units'], width=quiver['width'], color=quiver['color'],
                     headwidth=quiver['headwidth'], scale=quiver['scale'], headlength=quiver['headlength'], headaxislength=quiver['headaxislength'], visible=True)
    from scipy.interpolate import griddata
    if stream:
        u_stream = stream['u'][lat0_i:latf_i,lon0_i:lonf_i]
        v_stream = stream['v'][lat0_i:latf_i,lon0_i:lonf_i]
        x_stream = np.linspace(PLT_LON.min(), PLT_LON.max(), np.shape(PLT_LON)[0])
        y_stream = np.linspace(PLT_LAT.min(), PLT_LAT.max(), np.shape(PLT_LAT)[1])
        print(PLT_LON.min(), PLT_LON.max())
        print(PLT_LAT.min(), PLT_LAT.max())
        xi, yi = np.meshgrid(x_stream,y_stream)
        #then, interpolate your data onto this grid:
        px = PLT_LON.flatten()
        py = PLT_LAT.flatten()
        gu = griddata(np.r_[ px[None,:], py[None,:] ].T, u_stream.flatten(), (xi,yi))
        gv = griddata(np.r_[ px[None,:], py[None,:] ].T, v_stream.flatten(), (xi,yi))
        strm = ax.streamplot(x_stream, y_stream, gu, gv, density=stream['density'], linewidth=stream['linewidth'], color=stream['color'])
        
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

    if adjust_plt : ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    if method == 'contourf' or method == 'filled_contour':
        e = csf
    elif method == 'pcolor' or method == 'contour_pcolor':
        e = pc
    else:
        e = cs
    return e
