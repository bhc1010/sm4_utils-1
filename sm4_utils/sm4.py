import os, datetime
import spym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from pathlib import Path
from enum import Enum

class SM4:
    FileType = Enum("FileType", ["Image", "dIdV", "IZ"])
    Topography = Enum("Topography", ["Forward", "Backward"])
    def __init__(self, src_path: str):
        self.path = Path(src_path)
        self.fname = self.path.stem
        self.raw = spym.load(src_path)
        self.fig = None
        self.type = None

        if self.raw is None:
            print("There was a problem loading the file. Ensure that the full path to the file is correct.")
            return

        if 'Current' in self.raw.data_vars:
            match self.raw.Current.RHK_LineTypeName:
                case 'RHK_LINE_IV_SPECTRUM':
                    self.type = SM4.FileType.dIdV
                case 'RHK_LINE_IZ_SPECTRUM':
                    self.type = SM4.FileType.IZ
        elif 'Topography_Forward' in self.raw.data_vars:
            self.type = SM4.FileType.Image

    def scale_data(self, scale: float):
        match self.type:
            case SM4.FileType.Image:
                print('Cannot scale image data. Access self.raw to scale data manually.')
            case SM4.FileType.dIdV:
                self.raw.LIA_Current.data *= scale
            case SM4.FileType.IZ:
                self.raw.Current.data *= scale

    def minimum(self, image: Topography = None) -> float:
        match self.type:
            case SM4.FileType.Image:
                match image:
                    case SM4.Topography.Forward:
                        return np.min(self.raw.Topography_Forward.data)
                    case SM4.Topography.Backward:
                        return np.min(self.raw.Topography_Backward.data)
                    case _:
                        print("Indicate which image to use with the image parameter. (e.g. image=SM4.Topography.Forward)")
                        return None
            case SM4.FileType.dIdV:
                return np.min(self.raw.LIA_Current.data)
            case SM4.FileType.IZ:
                return np.min(self.raw.Current.data)

    def maximum(self, image: Topography = None) -> float:
        match self.type:
            case SM4.FileType.Image:
                match image:
                    case SM4.Topography.Forward:
                        return np.max(self.raw.Topography_Forward.data)
                    case SM4.Topography.Backward:
                        return np.max(self.raw.Topography_Backward.data)
                    case _:
                        print("Indicate which image to use with the image parameter. (e.g. image=SM4.Topography.Forward)")
                        return None
            case SM4.FileType.dIdV:
                return np.max(self.raw.LIA_Current.data)
            case SM4.FileType.IZ:
                return np.max(self.raw.Current.data)

    def plot_topo(self, 
                  image: Topography = None, 
                  align=True, 
                  plane=True, 
                  fix_zero=True, 
                  show_axis=False, 
                  figsize=(8,8), 
                  scalebar_height=None):
        if self.type is not SM4.FileType.Image:
            print("File has no real image data.")
            return

        if image is None:
            print("No image type given. Set image parameter to a SM4.Topography value. SM4.Topography can be either Forward or Backward. e.g.: my_sm4.plot_image(image=SM4.Topography.Forward).")
            return

        if image == SM4.Topography.Forward:
            img = self.raw.Topography_Forward
        elif image == SM4.Topography.Backward:
            img = self.raw.Topography_Backward
        else:
            print("Incorrect image type given. Something went wrong")
            return

        if align:
            img.spym.align()
        if plane:
            img.spym.plane()
        if fix_zero:
            img.spym.fixzero()
        
        fig, ax = plt.subplots(figsize=figsize)
        if not show_axis:
            ax.axis('off')
        else:
            ax.set_xlabel("[nm]")
            ax.set_ylabel("[nm]")

        size = round(img.RHK_Xsize * abs(img.RHK_Xscale) * 1e9, 3)
        if scalebar_height is None:
            scalebar_height = 0.01 * size
        fontprops = fm.FontProperties(size=14)
        scalebar = AnchoredSizeBar(ax.transData,
                                   size/5, f'{size/5} nm', 'lower left',
                                   pad=0.25,
                                   color='white',
                                   frameon=False,
                                   size_vertical = scalebar_height,
                                   offset=1,
                                   fontproperties=fontprops)

        img = ax.imshow(img.data, extent=[0, size, 0, size], cmap='afmhot')
        ax.add_artist(scalebar)

        fig.tight_layout()
        return (fig, ax)

    def plot_waterfall(self, 
                       figsize=(8,8), 
                       spacing=None, 
                       cmap=plt.cm.jet, 
                       show_hysteresis=False):
        match self.type:
            case SM4.FileType.dIdV:
                return self.plot_sts_waterfall(figsize=figsize, spacing=spacing, cmap=cmap)
            case SM4.FileType.IZ:
                return self.plot_iz_waterfall(figsize=figsize, spacing=spacing, cmap=cmap, show_hysteresis=show_hysteresis)
            case _:
                print("File has no spectral data.")
                return

    def plot_matrix(self, 
                    aspect=None, 
                    norm=None, 
                    cmap='jet', 
                    color_bar=True, 
                    vmax=None, 
                    clip_max=None,
                    clip_min=None,
                    fontsize=None, 
                    absolute_hysteresis=True):
        match self.type:
            case SM4.FileType.dIdV:
                return self.plot_sts_matrix(aspect=aspect, 
                                            norm=norm, 
                                            cmap=cmap, 
                                            color_bar=color_bar, 
                                            vmax=vmax, 
                                            clip_max=clip_max, 
                                            clip_min=clip_min, 
                                            fontsize=fontsize)
            case SM4.FileType.IZ:
                return self.plot_iz_matrix(aspect=aspect, 
                                           norm=norm, 
                                           cmap=cmap, 
                                           color_bar=color_bar, 
                                           vmax=vmax, 
                                           clip_max=clip_max, 
                                           clip_min=clip_min, 
                                           fontsize=fontsize, 
                                           absolute_hysteresis=absolute_hysteresis)
            case SM4.FileType.Image:
                print("File has no spectral data.")
                return
            case _:
                print("plot_matrix requires a valid file to be defined.")
                return

    def plot_coords(self,
                    image_path=None, 
                    image_type=None, 
                    align=True, 
                    plane=True, 
                    fix_zero=True, 
                    show_axis=False, 
                    scalebar_height=None, 
                    figsize=(8,8), 
                    cmap='afmhot',
                    arrow=False):
        args = locals()
        args.pop('self')
        match self.type:
            case SM4.FileType.dIdV:
                return self.plot_sts_coords(**args)
            case SM4.FileType.IZ:
                return self.plot_iz_coords(**args)

    def plot_sts_waterfall(self, 
                           figsize=(8,8), 
                           spacing=None, 
                           cmap=plt.cm.jet):
        if self.type is not SM4.FileType.dIdV:
            print("File contains no STS data.")
            return 
        
        ldos = self.raw.LIA_Current

        if 'RHK_SpecDrift_Xcoord' not in ldos.attrs:
            print('RHK_SpecDrift_Xcoord not in LIA_Current attributes.')
            return
        
        ldos_coords = self.unique_coordinates(zip(ldos.RHK_SpecDrift_Xcoord, ldos.RHK_SpecDrift_Ycoord))
        N = len(ldos_coords)
        if N == 0:
            print("No STS data found.")
            return

        xsize = ldos.RHK_Xsize
        total = ldos.RHK_Ysize
        repetitions = total//N
        x = ldos.LIA_Current_x.data * 1e3
        ldos_ave = ldos.data.reshape(xsize, N, repetitions).mean(axis=2).T

        ## Plot
        if spacing is None:
            spacing = np.max(ldos_ave) / 10
        waterfall_offset = np.flip([i * spacing for i in range(N)])
        colors = cmap(np.linspace(0, 1, N))

        fig, ax = plt.subplots(figsize=figsize)
        for (i, dIdV) in enumerate(ldos_ave):
            ax.plot(x, dIdV + waterfall_offset[i], c=colors[i])

        fig.tight_layout()
        return (fig, ax)

    def plot_sts_matrix(self, 
                        aspect=None, 
                        norm=None, 
                        cmap='jet', 
                        color_bar=True, 
                        vmax=None,
                        clip_max=None,
                        clip_min=None,
                        fontsize=None):
        if self.type is not SM4.FileType.dIdV:
            print("File contains no STS data.")
            return 
        
        ldos = self.raw.LIA_Current
        
        if 'RHK_SpecDrift_Xcoord' not in ldos.attrs:
            print('RHK_SpecDrift_Xcoord not in LIA_Current attributes.')
            return

        ldos_coords = self.unique_coordinates(zip(ldos.RHK_SpecDrift_Xcoord, ldos.RHK_SpecDrift_Ycoord))
        N = len(ldos_coords)
        if N == 0:
            print("No STS data found.")
            return

        xsize = ldos.RHK_Xsize
        total = ldos.RHK_Ysize
        repetitions = total//N
        x = ldos.LIA_Current_x.data * 1e3
        ldos_ave = ldos.data.reshape(xsize, N, repetitions).mean(axis=2).T

        fig, ax = plt.subplots(figsize=(16,7))

        line_cut = (ldos_coords[-1][0] - ldos_coords[0][0], ldos_coords[-1][1] - ldos_coords[0][1])
        line_length = np.sqrt(line_cut[0]**2 + line_cut[1]**2) * 1e9

        if aspect is None:
            aspect = abs(x[-1] - x[0]) / line_length

        if clip_max:
            ldos_ave = ldos_ave.clip(None, clip_max)
        if clip_min:
            ldos_ave = ldos_ave.clip(clip_min, None)

        img = ax.imshow(ldos_ave, aspect=aspect, extent=[x[0], x[-1], 0, line_length], norm=norm, cmap=cmap, vmax=vmax)

        ax.set_yticks(np.linspace(line_length, 0, 10))
        ax.set_yticklabels(map(lambda x: "%.2f" % x, np.linspace(line_length, 0, 10)))

        ax.set_xlabel("Voltage (mV)", fontsize=fontsize)
        ax.set_ylabel("Distance (nm)", fontsize=fontsize)

        if color_bar:
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            cb = fig.colorbar(img, cax=cax)
            return (fig, ax, cb)
        else:
            fig.tight_layout()
            return (fig, ax, None)
    
    def plot_sts_coords(self, 
                    image_path=None, 
                    image_type=None, 
                    align=True, 
                    plane=True, 
                    fix_zero=True, 
                    show_axis=False, 
                    scalebar_height=None, 
                    figsize=(8,8), 
                    cmap='afmhot',
                    arrow=False):
        if self.type is not SM4.FileType.dIdV:
            print("File contains no STS data.")
            return 
        
        ldos = self.raw.LIA_Current

        if 'RHK_SpecDrift_Xcoord' not in ldos.attrs:
            print('RHK_SpecDrift_Xcoord not in LIA_Current attributes.')
            return
        
        ldos_coords = self.unique_coordinates(zip(ldos.RHK_SpecDrift_Xcoord, ldos.RHK_SpecDrift_Ycoord))
        N = len(ldos_coords)
        if N == 0:
            print("No STS data found.")
            return

        topo = None
        if image_path is None:
            topo = self.get_last_image()
        else:
            try:
                topo_sm4 = spym.load(image_path)
                if image_type is SM4.Topography.Backward:
                    topo = topo_sm4.Topography_Backward
                elif image_type is SM4.Topography.Forward or image_type is None:
                    topo = topo_sm4.Topography_Forward
                else:
                    print("Invalid image type.")
            except:
                print(f"Couldn't load topography data from {image_path}")
                return
        
        if topo is None:
            print("No topography data found.")
            return

         ## Spec Coordinates
        xoffset = topo.RHK_Xoffset
        yoffset = topo.RHK_Yoffset
        xscale = topo.RHK_Xscale
        yscale = topo.RHK_Yscale
        xsize = topo.RHK_Xsize
        ysize = topo.RHK_Ysize
        width = np.abs(xscale * xsize)
        height = np.abs(yscale * ysize)

        offset = np.array([xoffset, yoffset]) + 0.5 * np.array([-width, -height])
        colors = plt.cm.jet(np.linspace(0, 1, N))
        
        fig, ax = plt.subplots(figsize=figsize)
        if not show_axis:
            ax.axis('off')
        else:
            ax.set_xlabel("[nm]")
            ax.set_ylabel("[nm]")

        if align:
            topo.spym.align()
        if plane:
            topo.spym.plane()
        if fix_zero:
            topo.spym.fixzero()

        size = round(topo.RHK_Xsize * abs(topo.RHK_Xscale) * 1e9, 3)
        ax.imshow(topo.data, extent=[0, size, 0, size], cmap=cmap)

        if not arrow:
            for (i, real_coord) in enumerate(ldos_coords):
                view_coord = np.array(real_coord - offset) * 1e9
                ax.plot(view_coord[0], view_coord[1], marker="o", c=colors[i])
        else:
            (x1, y1) = np.array(ldos_coords[0] - offset) * 1e9
            (x2, y2) = np.array(ldos_coords[-1] - offset) * 1e9
            ax.arrow(x1, y1, x2 - x1, y2 - y1, lw=0.1, width=0.2, length_includes_head=True, edgecolor='w', facecolor='w')

        size = round(topo.RHK_Xsize * abs(topo.RHK_Xscale) * 1e9, 3)
        if scalebar_height is None:
            scalebar_height = 0.01 * size
        fontprops = fm.FontProperties(size=14)
        scalebar = AnchoredSizeBar(ax.transData,
                                   size/5, f'{size/5} nm', 'lower left',
                                   pad=0.25,
                                   color='white',
                                   frameon=False,
                                   size_vertical = scalebar_height,
                                   offset=1,
                                   fontproperties=fontprops)

        ax.add_artist(scalebar)

        fig.tight_layout()
        return (fig, ax)

    def plot_iz_waterfall(self, 
                          figsize=(8,8), 
                          spacing=None, 
                          cmap=plt.cm.jet, 
                          show_hysteresis=False):
        if self.type is not SM4.FileType.IZ:
            print("File contains no STS data.")
            return
        
        iz = self.raw.Current
        
        if 'RHK_SpecDrift_Xcoord' not in iz.attrs:
            print('RHK_SpecDrift_Xcoord not in Current attributes.')
            return

        coords = self.unique_coordinates(zip(iz.RHK_SpecDrift_Xcoord, iz.RHK_SpecDrift_Ycoord))
        N = len(coords)
        if N == 0:
            print("No STS data found.")
            return

        xsize = iz.RHK_Xsize
        total = iz.RHK_Ysize
        repetitions = total//N
        x = iz.Current_x.data * 1e3
        iz_data = iz.data.reshape(xsize, N, repetitions).T
        approach = iz_data[::2, :].mean(axis=0)
        retract = iz_data[1::2, :].mean(axis=0)

        ## Plot
        if spacing is None:
            spacing = np.max(iz_data) / 10
        waterfall_offset = np.flip([i * spacing for i in range(N)])
        colors = cmap(np.linspace(0, 1, N))

        fig, ax = plt.subplots(figsize=figsize)
        for (i, (appr, retr)) in enumerate(zip(approach, retract)):
            if show_hysteresis:
                ax.plot(x, retr - appr + waterfall_offset[i], c=colors[i])
            else:
                ax.plot(x, appr + waterfall_offset[i], c=colors[i])
                ax.plot(x, retr + waterfall_offset[i], c=colors[i], ls='--')

        fig.tight_layout()
        return (fig, ax)

    def plot_iz_matrix(self, 
                       aspect=None, 
                       norm=None, 
                       cmap='jet', 
                       color_bar=True, 
                       vmax=None,
                       clip_min=None,
                       clip_max=None,
                       fontsize=None, 
                       absolute_hysteresis=True):
        if self.type is not SM4.FileType.IZ:
            print("File contains no IZ data.")
            return 
        
        iz = self.raw.Current
        
        if 'RHK_SpecDrift_Xcoord' not in iz.attrs:
            print('RHK_SpecDrift_Xcoord not in Current attributes.')
            return

        coords = self.unique_coordinates(zip(iz.RHK_SpecDrift_Xcoord, iz.RHK_SpecDrift_Ycoord))
        N = len(coords)
        if N == 0:
            print("No STS data found.")
            return

        xsize = iz.RHK_Xsize
        total = iz.RHK_Ysize
        repetitions = total//N
        x = iz.Current_x.data * 1e9
        iz_data = iz.data.reshape(xsize, N, repetitions).T
        approach = iz_data[::2, :].mean(axis=0)
        retract = iz_data[1::2, :].mean(axis=0)
        iz_ave = np.flip(retract - approach)

        if absolute_hysteresis:
            iz_ave = np.abs(iz_ave)

        fig, ax = plt.subplots(figsize=(16,7))

        line_cut = (coords[-1][0] - coords[0][0], coords[-1][1] - coords[0][1])
        line_length = np.sqrt(line_cut[0]**2 + line_cut[1]**2) * 1e9

        if aspect is None:
            aspect = abs(x[0]) / line_length

        if clip_max:
            iz_ave = iz_ave.clip(None, clip_max)
        if clip_min:
            iz_ave = iz_ave.clip(clip_min, None)

        img = ax.imshow(iz_ave, aspect=aspect, extent=[x[0], x[-1], line_length, 0], norm=norm, cmap=cmap, vmax=vmax)

        ax.set_yticks(np.linspace(0, line_length, 10))
        ax.set_yticklabels(map(lambda x: "%.2f" % x, np.linspace(line_length, 0, 10)))
        ax.set_xlabel("Tip height (nm)", fontsize=fontsize)
        ax.set_ylabel("Distance (nm)", fontsize=fontsize)

        if color_bar:
            cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
            cb = fig.colorbar(img, cax=cax)
            return (fig, ax, cb)
        else:
            fig.tight_layout()
            return (fig, ax, None)       

    def plot_iz_coords(self, 
                    image_path=None, 
                    image_type=None, 
                    align=True, 
                    plane=True, 
                    fix_zero=True, 
                    show_axis=False, 
                    scalebar_height=None, 
                    figsize=(8,8), 
                    cmap='afmhot',
                    arrow=False):
        if self.type is SM4.FileType.dIdV:
            print("File contains no STS data.")
            return 
        
        iz = self.raw.Current

        if 'RHK_SpecDrift_Xcoord' not in iz.attrs:
            print('RHK_SpecDrift_Xcoord not in LIA_Current attributes.')
            return
        
        coords = self.unique_coordinates(zip(iz.RHK_SpecDrift_Xcoord, iz.RHK_SpecDrift_Ycoord))
        N = len(coords)
        if N == 0:
            print("No STS data found.")
            return

        topo = None
        if image_path is None:
            topo = self.get_last_image()
        else:
            try:
                topo_sm4 = spym.load(image_path)
                if image_type is SM4.Topography.Backward:
                    topo = topo_sm4.Topography_Backward
                elif image_type is SM4.Topography.Forward or image_type is None:
                    topo = topo_sm4.Topography_Forward
                else:
                    print("Invalid image type.")
            except:
                print(f"Couldn't load topography data from {image_path}")
                return
        
        if topo is None:
            print("No topography data found.")
            return

         ## Spec Coordinates
        xoffset = topo.RHK_Xoffset
        yoffset = topo.RHK_Yoffset
        xscale = topo.RHK_Xscale
        yscale = topo.RHK_Yscale
        xsize = topo.RHK_Xsize
        ysize = topo.RHK_Ysize
        width = np.abs(xscale * xsize)
        height = np.abs(yscale * ysize)

        offset = np.array([xoffset, yoffset]) + 0.5 * np.array([-width, -height])
        colors = plt.cm.jet(np.linspace(0, 1, N))
        
        fig, ax = plt.subplots(figsize=figsize)
        if not show_axis:
            ax.axis('off')
        else:
            ax.set_xlabel("[nm]")
            ax.set_ylabel("[nm]")

        if align:
            topo.spym.align()
        if plane:
            topo.spym.plane()
        if fix_zero:
            topo.spym.fixzero()

        size = round(topo.RHK_Xsize * abs(topo.RHK_Xscale) * 1e9, 3)
        ax.imshow(topo.data, extent=[0, size, 0, size], cmap=cmap)

        if not arrow:
            for (i, real_coord) in enumerate(coords):
                view_coord = np.array(real_coord - offset) * 1e9
                ax.plot(view_coord[0], view_coord[1], marker="o", c=colors[i])
        else:
            (x1, y1) = np.array(coords[0] - offset) * 1e9
            (x2, y2) = np.array(coords[-1] - offset) * 1e9
            ax.arrow(x1, y1, x2 - x1, y2 - y1, lw=0.1, width=0.2, length_includes_head=True, edgecolor='w', facecolor='w')

        size = round(topo.RHK_Xsize * abs(topo.RHK_Xscale) * 1e9, 3)
        if scalebar_height is None:
            scalebar_height = 0.01 * size
        fontprops = fm.FontProperties(size=14)
        scalebar = AnchoredSizeBar(ax.transData,
                                   size/5, f'{size/5} nm', 'lower left',
                                   pad=0.25,
                                   color='white',
                                   frameon=False,
                                   size_vertical = scalebar_height,
                                   offset=1,
                                   fontproperties=fontprops)

        ax.add_artist(scalebar)

        fig.tight_layout()
        return (fig, ax)

    def get_last_image(self):
        if len(self.path.name.split("_")) < 7:
            return None
        
        src_dir = self.path.parent
        files = [x for x in os.listdir(src_dir) if x.endswith('.sm4')]
        dates = [x.split('.')[0].split('_') for x in files]
        dates = [x[-7:] for x in dates if len(x) > 7]
        dates = [datetime.datetime(*[int(d) for d in date]) for date in dates]
        dates = list(zip(dates, range(len(dates))))
        dates_sorted, permuted_indices = list(zip(*sorted(dates)))
        file_date = self.path.name.split('.')[0].split('_')[-7:]  # Date of the current file
        file_date = datetime.datetime(*[int(d) for d in file_date])
        
        files = [files[i] for i in list(permuted_indices)]
        idx = dates_sorted.index(file_date) # index of the current file in the date ordered list
        topography = None

        while idx >= 0:
            f = spym.load(os.path.join(src_dir, files[idx]))
            if f is None:
                idx -= 1
            elif 'data_vars' in f.__dir__():
                if 'Topography_Forward' in f.data_vars:
                    topography = f.Topography_Forward
                    if topography.data.shape[0] == topography.data.shape[1]: ### There is no full proof way to tell the difference between data that has only dIdV and data that has both image and dIdV - checking if the image is square is the closest option
                        line_average = np.average(topography.data, axis=1)
                        num_zeros = len(topography.data) - np.count_nonzero(line_average)
                        if num_zeros == 0:
                            break
                        else:
                            topography = None
                idx -= 1
            else:
                idx -= 1

        return topography

    def unique_coordinates(self, coords):
        seen = set()
        seen_add = seen.add
        return [x for x in coords if not (x in seen or seen_add(x))]