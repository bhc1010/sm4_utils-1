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
        self.file = spym.load(src_path)
        self.fig = None
        self.type = None

        if 'Current' in self.file.data_vars:
            match self.file.Current.RHK_LineTypeName:
                case 'RHK_LINE_IV_SPECTRUM':
                    self.type = SM4.FileType.dIdV
                case 'RHK_LINE_IZ_SPECTRUM':
                    self.type = SM4.FileType.IZ
        elif 'Topography_Forward' in self.file.data_vars:
            self.type = SM4.FileType.Image

    def plot_topo(self, image: Topography = None, align=True, plane=True, fix_zero=True, show_axis=False, figsize=(8,8), scalebar_height=None):
        if self.type is not SM4.FileType.Image:
            print("File has no real image data.")
            return

        if image is None:
            print("No image type given. Set image parameter to a SM4.Topography value. SM4.Topography can be either Forward or Backward. e.g.: my_sm4.plot_image(image=SM4.Topography.Forward).")
            return

        if image == SM4.Topography.Forward:
            img = self.file.Topography_Forward
        elif image == SM4.Topography.Backward:
            img = self.file.Topography_Backward
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

        return (fig, ax)

    def plot_waterfall(self, figsize=(8,8), skip=None, cmap=plt.cm.jet):
        if self.type is not SM4.FileType.dIdV:
            print("File contains no STS data.")
            return 
        
        ldos = self.file.LIA_Current

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
        if skip is None:
            skip = np.max(ldos_ave) / 10
        waterfall_offset = np.flip([i * skip for i in range(N)])
        colors = cmap(np.linspace(0, 1, N))

        fig, ax = plt.subplots(figsize=figsize)
        for (i, dIdV) in enumerate(ldos_ave):
            ax.plot(x, dIdV + waterfall_offset[i], c=colors[i])

        return (fig, ax)

    def plot_spectral_cut(self):
        pass

    def plot_waterfall_with_image(self, figsize=(16,8), image_path=None, image_type=None, align=True, plane=True, fix_zero=True, show_axis=False, scalebar_height=None):
        if self.type is not SM4.FileType.dIdV:
            print("File contains no STS data.")
            return 
        
        ldos = self.file.LIA_Current

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
            
        xsize = ldos.RHK_Xsize
        total = ldos.RHK_Ysize
        repetitions = total//N
        x = ldos.LIA_Current_x.data * 1e3
        ldos_ave = ldos.data.reshape(xsize, N, repetitions).mean(axis=2).T

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

        ## Plot
        skip = np.max(ldos_ave) / 10
        waterfall_offset = np.flip([i * skip for i in range(N)])
        colors = plt.cm.jet(np.linspace(0, 1, N))
        
        fig, ax = plt.subplots(figsize=figsize, ncols=2)
        image_ax, spec_ax = ax
        if not show_axis:
            image_ax.axis('off')
        else:
            image_ax.set_xlabel("[nm]")
            image_ax.set_ylabel("[nm]")

        if align:
            topo.spym.align()
        if plane:
            topo.spym.plane()
        if fix_zero:
            topo.spym.fixzero()

        size = round(topo.RHK_Xsize * abs(topo.RHK_Xscale) * 1e9, 3)
        image_ax.imshow(topo.data, extent=[0, size, 0, size], cmap='afmhot')

        for (i, (dIdV, real_coord)) in enumerate(zip(ldos_ave, ldos_coords)):
            view_coord = np.array(real_coord - offset) * 1e9

            spec_ax.plot(x, dIdV + waterfall_offset[i], c=colors[i])
            image_ax.plot(view_coord[0], view_coord[1], marker="o", c=colors[i])

        return (fig, ax)
    
    def plot_sts_coords(self, image_path=None, image_type=None, align=True, plane=True, fix_zero=True, show_axis=False, scalebar_height=None, figsize=(8,8), cmap='afmhot'):
        if self.type is not SM4.FileType.dIdV:
            print("File contains no STS data.")
            return 
        
        ldos = self.file.LIA_Current

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

        for (i, real_coord) in enumerate(ldos_coords):
            view_coord = np.array(real_coord - offset) * 1e9
            ax.plot(view_coord[0], view_coord[1], marker="o", c=colors[i])

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