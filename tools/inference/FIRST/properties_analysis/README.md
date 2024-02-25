# Statistical properties code for FIRST data
- **cutout_provider_core-racs:** Grabbing cutouts from various surveys: FIRST, VLASS, NVSS, SDSS, et al. This Cutout Core is built on the cutout core command-line tool [cutout_provider_core](https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/cirada/data/cutout_provider_core-racs.zip).
- **NED:** Searching optical/infrared from NASA/IPAC Extragalactic Database (NED).
- **Position_angle:** Calculate radio position angles and analyze their alignment.
- **Radio_Luminosity:** Calculate and analyze radio luminosity at 1.4GHz.
- **linear_size:** Calculate the largest angular size (LAS) and largest linear size (LLS).
- **optical_spectra:** Query and analyze SDSS spectral data.
- **radio_optical_overlay:** Plot radio image overlaid on SDSS image.
- **spectral_indexes:** Calculation and analysis of radio spectral index between 1.4 GHz and 3 GHz.
- **total_flux_density:** Calculate integrated or total flux density for 1.4 GHz and 3 GHz data.

We have used this code to build a FR-II Radio Galaxy Catalog ([FRIIRGcat](https://drive.google.com/file/d/19m_ma-2fFIWVZ8WJphXxr5W_HXkyIAeX/view?usp=drive_link)), which contains 45,241 candidates. If you benefit from FRIIRGcat, please cite the [FRIIRGcat paper](https://ui.adsabs.harvard.edu/abs/2024arXiv240108048L/abstract):
```
@ARTICLE{2024arXiv240108048L,
       author = {{Lao}, Bao-Qiang and {Yang}, Xiao-Long and {Jaiswal}, Sumit and {Mohan}, Prashanth and {Sun}, Xiao-Hui and {Qin}, Sheng-Li and {Zhao}, Ru-Shuang},
        title = "{A Machine Learning made Catalog of FR II Radio Galaxies from the FIRST Survey}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Astrophysics of Galaxies},
         year = 2024,
        month = jan,
          eid = {arXiv:2401.08048},
        pages = {arXiv:2401.08048},
          doi = {10.48550/arXiv.2401.08048},
archivePrefix = {arXiv},
       eprint = {2401.08048},
 primaryClass = {astro-ph.GA},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv240108048L},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

# Step by step to build FRIIRGcat:
```
1. cd ./total_flux_density && ./fix_FIRST_flux_fr2.sh
2. cd ./total_flux_density && python3 vlass_fr2.py
3. cd ./linear_size && python3 LAS.py
4. cd ./total_flux_density && python3 nvss_fr2_all.py
5. cd ./total_flux_density && python3 final_flux.py
6. cd ./spectral_indexes && python3 spix.py
7. cd ./Position_angle && python3 RPA_new.py
8. cd ./final_catalog && python3 generate_catalog_fr2.py
9. cd ./optical_spectra && ./optical_spectra.sh
9. cd ./final_catalog && python3 final_fr2.py
```


# Step by step to build HTRGcat:
```
1. cd ./total_flux_density && ./fix_FIRST_flux_ht.sh
2. cd ./total_flux_density && python3 vlass_ht.py
3. cd ./total_flux_density && python3 nvss_ht_all.py
4. cd ./total_flux_density && python3 final_flux_ht.py
TBD ...
```
