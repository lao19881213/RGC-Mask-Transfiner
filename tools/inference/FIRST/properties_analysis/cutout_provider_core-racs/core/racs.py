from unicodedata import name
import urllib

from astropy import units as u
from astroquery.cadc import Cadc
from astroquery.utils.tap.core import TapPlus
from .survey_abc import SurveyABC
from .toolbox import pad_string_lines
from .survey_filters import racs_filters


class RACS(SurveyABC):
    def __init__(self, filter=None):
        super().__init__()
        self.needs_trimming = False
        self.filter = filter
        self.filenames = []

    @staticmethod
    def get_supported_filters():
        return racs_filters

    @staticmethod
    def get_epoch(fileOrURL):
        url_match  = fileOrURL.split('RACS-DR')[1][0]
        file_match = fileOrURL.split('RACS__-DR')[1][0]
        return url_match if url_match else file_match

    @staticmethod
    def get_cutout_url(ql_url, coords, radius):
        standard_front = 'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/sync?ID=casda%3ARACS%2F'
        encoded_ql = urllib.parse.quote(ql_url.split("/")[-1])
        encoded_ql = encoded_ql.replace('%3F', '&').replace('?', '&')
        cutout_end = f"&CIRCLE={coords.ra.value}+{coords.dec.value}+{radius.value}"
        return standard_front + encoded_ql + cutout_end

    def cone_search(self, coords, radius):
        # Catalog URLs that contain RACS data
        # https://research.csiro.au/casda/the-rapid-askap-continuum-survey-stokes-i-source-catalogue-data-release-1/
        urls = {"gaussian_cuts": "racs_dr1_gaussians_galacticcut_v2021_08_v01",
                "gaussian_regions": "racs_dr1_gaussians_galacticregion_v2021_08_v01",
                "sources_cuts": "racs_dr1_sources_galacticcut_v2021_08_v01",
                "sources_regions": "racs_dr1_sources_galacticregion_v2021_08_v01"}

        # Calculating galactic latitude (b) to determine which URLs to access
        galactic_latitude = coords.galactic.b

        # If (b) is less than 4, use regions URLs. If greater than 6, use cuts URLs. Otherwise, use all URLs.
        # https://cirada.slack.com/archives/G012UCGDB1V/p1651706293317779?thread_ts=1651705633.851679&cid=G012UCGDB1V
        if abs(galactic_latitude) <= 4:
            del urls['gaussian_cuts']
            del urls['sources_cuts']
        elif abs(galactic_latitude) > 6:
            del urls['gaussian_regions']
            del urls['sources_regions']
            
        # https://astroquery.readthedocs.io/en/latest/utils/tap.html
        # https://astroquery.readthedocs.io/en/v0.3.9/api/astroquery.gaia.GaiaClass.html#astroquery.gaia.GaiaClass.launch_job_async
        job_list = []
        for source in urls:
            query = f"SELECT * FROM AS110.{urls[source]} where 1=CONTAINS(POINT('ICRS', ra, dec),CIRCLE('ICRS',{coords.ra.value},{coords.dec.value},{radius.value}))"
            casdatap = TapPlus(url="https://casda.csiro.au/casda_vo_tools/tap")
            job = casdatap.launch_job_async(query, dump_to_file=True, output_file=self.out_dir, output_format="fits")
            job_list.append(job)

        # TODO overwrite SurveyABC.get_cutout and SurveyABC.
        return job_list

    def get_filter_setting(self):
        return self.filter

    def add_cutout_service_comment(self, hdu):
        hdu.header.add_comment(pad_string_lines("Quick Look images do not fully sample the PSF, and are cleaned to a "
                                                "threshold of ~5 sigma (details can be found in the weblogs for "
                                                "individual images). They are used for Quality Assurance and for "
                                                "transient searches, but should not be used for any other purpose. In "
                                                "addition to imaging artifacts, source positions can be off by up to "
                                                "1-arcsec, and the flux density uncertainties are ~10-20%. \
                   "), after=-1)
        hdu.header.add_comment(pad_string_lines("The direct data service at CADC was used to provide this cutout: "
                                                " (https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/doc/data/) \
                                "), after=-1)

    def add_filename(self, ql_url):
        filename = urllib.parse.quote(ql_url.split("/")[-1])
        filename = filename.replace('%3F', '&').replace('%2B', '+').replace('?', '&')
        self.filenames.append(filename)
    
    def get_tile_urls(self, position, size):
        cadc = Cadc()
        radius = (size/2.0).to(u.deg)
        urls = []

        all_rows = cadc.exec_sync(f"Select Plane.publisherID, Observation.requirements_flag FROM caom2.Plane AS Plane JOIN caom2.Observation AS Observation \
                                    ON Plane.obsID = Observation.obsID WHERE  ( Observation.collection = 'RACS' \
                                    AND INTERSECTS( CIRCLE('ICRS', {position.ra.value}, {position.dec.value},  {radius.value}), Plane.position_bounds ) = 1) \
                                    AND ( Observation.requirements_flag IS NULL OR Observation.requirements_flag != 'fail') ")

        if len(all_rows) > 0:
            ql_urls = cadc.get_data_urls(all_rows)
            if ql_urls:
                for url in ql_urls:
                    cutout_url = RACS.get_cutout_url(url, position, radius)
                    urls.append(cutout_url)
                    self.add_filename(url)

        ### If adding any filters in then this is where would do it!!!#####
        #### e.g. filtered_results = results[results['time_exposure'] > 120.0] #####

        # if len(results) == 0:
        #     return list()
        # print("or this one?")
        # urls = cadc.get_image_list(results, position, radius)
        # if len(urls)==0:
        #     self.print("Cannot find {position.to_string('hmsdms')}, perhaps this hasn't been covered by RACS.")
        if self.filter:
            final_urls = []
            for url in urls:

                # TODO filter on epoch and frequency
                #   epoch is single digit as a string
                #   frequency is a range (how will user enter freq? low/mid/high or a float or a range of floats?)
                #   racs_filters are tuples like ("epoch", "frequency range")

                epoch = RACS.get_epoch(url)
                freq = "743.5-1031.5"
                self.print(f"FILTER: {self.filter}, VALUE: {self.filter.value}")

                if (epoch, freq) == self.filter.value:
                    final_urls.append(url)
                    
            urls = final_urls

        if len(urls) == 0:
            self.print(f"Cannot find ({position.to_string('hmsdms')}), perhaps this hasn't been covered by RACS.")
            return list()

        return urls

    def get_fits_header_updates(self, header, all_headers=None):
        # complex file name - extract from header info
        # fpartkeys = [f'FILNAM{i+1:02}' for i in range(12)]
        # nameparts = [header[key] for key in fpartkeys]

        # # create single string - FILNAM12 goes after a constant
        # vfile = nameparts[0]
        # for i in range(len(nameparts)-2):
        #     vfile = vfile + '.' + nameparts[i+1]

        # vfile = vfile + '.pbcor.' + nameparts[len(nameparts)-1] + '.subim.fits'
        print()

        header_updates = {
            'BAND': ('2-4 GHz', 'Frequency coverage of observation'),
            'RADESYS':  (header['RADESYS'], 'Coordinate system used'),
            'DATE-OBS': (header['DATE-OBS'], 'Obs. date'),
            'BUNIT': ('Jy/beam', 'Pixel flux unit'),
            'BMAJ':  (header['BMAJ'], 'Beam major axis [deg]'),
            'BMIN':  (header['BMIN'], 'Beam minor axis [deg]'),
            'BPA':   (header['BPA'], 'Beam position angle'),
            'BTYPE': (header['BTYPE'], 'Stokes polarisation')
            # TODO (Issue #6): Tiling issue and based on quick-look images -- I think...
            # 'IMFILE': (vfile, 'RACS image file'),
        }

        # Adds the filenames for the individual tiles to the PrimaryHDU of the fits file
        # NOTE: Individual HDUs within the FITS file will also contain their corresponding filename
        #       under the header 'FILNAM'
        for i, filename in enumerate(self.filenames):
            header_updates[f'FILNAM{i+1:02}'] = filename

        # # ONLY FOR MOSAICKED. complex file name list all originals gone into mosaic
        # if all_headers:
        #     for i in len(all_headers):
        #         fpartkeys = self.filenames.keys()

        #         nameparts = [header_updates[key] for key in fpartkeys]

        #         header_updates['IMFILE'+str(i+1).zfill(2)
        #                        ] = '.'.join(nameparts) + '.subim.fits'
        #         # create single string - FILNAM12 goes after a constant

        return header_updates
