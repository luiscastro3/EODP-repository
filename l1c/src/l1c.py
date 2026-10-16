
# LEVEL-1C

from l1c.src.initL1c import initL1c
from common.io.writeToa import writeToa, readToa
from common.io.readGeodetic import readGeodetic, getCorners
import mgrs
import numpy as np
from scipy.interpolate import bisplrep, bisplev
import matplotlib.pyplot as plt
from common.io.l1cProduct import writeL1c
from matplotlib import cm
from geopy.distance import great_circle

class l1c(initL1c):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1C Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the L1B in Radiances
            # -------------------------------------------------------------------------------
            toa = readToa(self.l1bdir, self.globalConfig.l1b_toa + band + '.nc')
            lat,lon = readGeodetic(self.gmdir, self.globalConfig.gm_geoloc)
            self.checkSize(lat,toa)

            # L1C reprojection onto the MGRS grid
            # -------------------------------------------------------------------------------
            lat_l1c, lon_l1c, toa_l1c = self.l1cProjtoa(lat, lon, toa, band)

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeL1c(self.outdir, self.globalConfig.l1c_toa + band, lat_l1c, lon_l1c, toa_l1c)

            # Generate and save report plots
            self.generate_report_plots(lat, lon, lat_l1c, lon_l1c, band)

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1C Module!")


    def l1cProjtoa(self, lat, lon, toa, band):
        '''
        This function reprojects the L1B radiances into the MGRS grid.

        The MGRS reference system
        https://www.bluemarblegeo.com/knowledgebase/calculator-2020/Military_Grid_Reference_System_(MGRS).htm
        MGRS: '31REQ4367374067'
        31 is the UTM zone, R is the UTM latitude band; EQ are the MGRS column and row band letters
        43673 is the MGRS Easting (5 dig); 74067 is the MGRS Northing (5dig)

        Python mgrs library documentation
        https://pypi.org/project/mgrs/

        :param lat: L1B latitudes [deg]
        :param lon: L1B longitudes [deg]
        :param toa: L1B radiances
        :param band: band
        :return: L1C radiances, L1C latitude and longitude in degrees
        '''
        #TODO

        tck = bisplrep(lat, lon, toa)
        m = mgrs.MGRS()
        mgrs_tiles = set([])

        for i in range(toa.shape[0]):
            for j in range(toa.shape[1]):
                mgrs_tile = m.toMGRS(lat[i,j],lon[i,j],MGRSPrecision=self.l1cConfig.mgrs_tile_precision)
                mgrs_tiles.add(mgrs_tile)

        mgrs_tiles = list(mgrs_tiles)

        toa_l1c = np.zeros(len(mgrs_tiles))
        lat_l1c = np.zeros(len(mgrs_tiles))
        lon_l1c = np.zeros(len(mgrs_tiles))

        for i in range(len(mgrs_tiles)):
            (lat_l1c[i], lon_l1c[i]) = m.toLatLon(mgrs_tiles[i])
            toa_l1c[i] = bisplev(lat_l1c[i], lon_l1c[i], tck)

        return lat_l1c, lon_l1c, toa_l1c

    def checkSize(self, lat,toa):
        '''
        Check the sizes of the input radiances and geodetic coordinates.
        If they don't match, exit.
        :param lat: Latitude 2D matrix
        :param toa: Radiance 2D matrix
        :return: NA
        '''
        #TODO

    def generate_report_plots(self, lat, lon, lat_l1c, lon_l1c, band):
        import os

        # Plot 1: L1B vs L1C Grid
        plt.figure(figsize=(10, 6))

        # Extract the boundary of the L1B grid to draw it as a continuous blue line
        lon_bound = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])
        lat_bound = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])

        plt.plot(lon_bound, lat_bound, c='blue', label='L1B Grid (Irregular)', linewidth=1.5)
        plt.scatter(lon_l1c, lat_l1c, c='red', label='L1C Grid (Regular MGRS)', s=5, alpha=0.6)

        plt.title(f'Projection on ground ({band})')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.outdir, f'figure_9_2_grids_{band}.png'), bbox_inches='tight')
        plt.close()

        # Plot 2: Spatial Sampling Distance
        central_row_idx = lat.shape[0] // 2
        central_lat = lat[central_row_idx, :]
        central_lon = lon[central_row_idx, :]

        distances = []
        for i in range(len(central_lat) - 1):
            point1 = (central_lat[i], central_lon[i])
            point2 = (central_lat[i + 1], central_lon[i + 1])
            dist_meters = great_circle(point1, point2).meters
            distances.append(dist_meters)

        plt.figure(figsize=(10, 4))
        plt.plot(distances, marker='o', linestyle='-', color='green')
        plt.title(f'Spatial Sampling Distance (Central Row of L1B - {band})')
        plt.xlabel('Pixel Index (Across-Track)')
        plt.ylabel('Distance (meters)')
        plt.grid(True)
        plt.savefig(os.path.join(self.outdir, f'spatial_sampling_distance_{band}.png'), bbox_inches='tight')
        plt.close()