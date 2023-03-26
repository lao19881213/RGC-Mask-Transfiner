<div id="top"></div>



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://cirada.ca/">
    <img src="logo_image/cirada_logo.png" alt="Logo" height="100">
  </a>
  <h1 align="center">Image Cutout Provider Core</h1>
  <br />
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#included-surveys">Included Surveys</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#common-installation-issues">Common Installation Issues</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#local-settings">Local Settings</a></li>
    <li><a href="#output">Output</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

The Image Cutout Provider Core allows astronomers to visualize data from multiple surveys at a given position in the sky. The project contains common software components that are applicable to all CIRADA-wide derived Cutout Software Products, which provide the following functionality:
* Fetching and saving cutout data
* 2D image preview
* Mosaicking: if the requested position and radius straddle boundaries in multiple FITS images for a given survey a mosaicked FITS file will be generated from all of these input images with each input image as an extension of the corresponding mosaicked FITS
* Grouping mosaicked FITS by observation date
* Filtering
* Trimming
* Error logging
* Processing status

The project includes a Command Line Interface with instructions listed below. There is also a public web service found at http://cutouts.cirada.ca

<p align="right">(<a href="#top">back to top</a>)</p>

### Included Surveys
| Survey | Band |
|---|---|
| VLASS| Radio|
| GLEAM | Radio |
| FIRST | Radio |
| NVSS | Radio|
| WISE |Infrared|
| PanSTARRS| Optical|
| SDSS| Optical|

### Built With

* Ubuntu 18.04 OS
* Python 3.6.8

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Below, we walk through the installation procedure for the Command Line Interface. Please install the prerequisites prior to installing the project.

### Prerequisites

You will need to install <a target=_blank href="https://montage-wrapper.readthedocs.io/en/v0.9.5/#installation">Montage</a>, which can be a little tricky. There are two methods that you may choose to follow in order to install Montage:

**Montage Installation Method 1:**

1. Download `http://montage.ipac.caltech.edu/download/Montage_v5.0.tar.gz` directly from <a target=_blank href="http://montage.ipac.caltech.edu/docs/download2.html">`http://montage.ipac.caltech.edu/docs/download2.html`</a>

   - *[Alternative]* On Ubuntu, you may run the following command to download the same file:
   ```bash
   sudo wget http://montage.ipac.caltech.edu/download/Montage_v5.0.tar.gz
   ```

2. Unpack your download:
   ```bash
   tar xvzf Montage_v5.0.tar.gz
   ```

3. Go into your newly unpacked download and build:
   ```bash
   cd Montage
   make
   ```

4. Ensure to add Montage to `$PATH` (here, we are adding Montage to a new directory `~/.montage` then adding to `$PATH`):
   ```bash
   cd ..
   mkdir ~/.montage
   mv Montage ~/.montage
   export PATH=$PATH:~/.montage/Montage/bin
   ```

5. *(Recommended)* Use **one** of the following commands to add `Montage` to your `$PATH` permanently:

   ```bash
   echo "PATH=$PATH:~/.montage/Montage/bin" >> ~/.zshrc
   echo "PATH=$PATH:~/.montage/Montage/bin" >> ~/.bash_profile
   ```

**Montage Installation Method 2:**

 * Command for Ubuntu OS:

   ```bash
   sudo apt-get install montage
   ```

 * Command for Mac OS:

   ```bash
   brew install montage
   ```

**To test:** run `mAdd` and you should see an output like the following, indicating that it is installed correctly:

```
[struct stat="ERROR", msg="Usage: mAdd [-d level] [-p imgdir] [-n(o-areas)] [-a mean|median|count] [-e(xact-size)] [-s statusfile] images.tbl template.hdr out.fits"]
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Installation

1. First clone this repo and `cd` into it. Then, create a virtual environment with `python virtualenv`:    

```bash
git clone https://gitlab.com/cirada/cutout_provider_core.git
cd cutout_provider_core
virtualenv -p python3 venv    
```

* *[Alternative]* You may instead choose to create a virtual environment with the following command:

```bash
python3 -m virtualenv venv
```

2. Activate your virtual environment:      

```bash
. venv/bin/activate    
```

3. Install all requirements from `requirements.txt` :   

```bash
# Downgrade setuptools because we are using legacy packages
pip3 install setuptools==45
pip3 install -r requirements.txt    
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Common Installation Issues

When installing the requirements from `requirements.txt`, you may run into the following issues:
* You may have problems with installing certain versions of modules (for instance, `pyvo` is notorious for installation problems). Feel free to take out the version requirements for such modules from `requirements.txt`.
* If you get an `astroquery` version error, you must install `astroquery` with the following commands:    
```bash
cd ..     
git clone https://github.com/astropy/astroquery.git        
cd astroquery    
python setup.py install
```
Then remove the astroquery line from `requirements.txt` and run this again:  
```bash
pip3 install -r requirements.txt
```

  * *[Alternative]* You may choose to directly install `astroquery` by:
```bash
pip3 install astroquery
```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

From the command line:  
``` bash 
python fetch_cutouts.py
```

with Commands:    
  `fetch        Single cutout fetching command.   `     
  `fetch_batch  Batch cutout fetching command.   `     

Options:   
```text
      -c, --coords TEXT     [required one of either -c or -n]    
      -n, --name TEXT
      -f, --file TEXT       batch file(s) name(s)  [required if fetch_batch]   
      -r, --radius INTEGER  [required]     
      -s, --surveys TEXT   
      -o, --output TEXT   
      -g, --groupby TEXT   
      -cf, --config TEXT   [optional]    
      --overwrite           overwrite existing duplicate target files (default
                            True)   
      --flush               flush existing target files (supersedes --overwrite)   
      --help                Show this message and exit.  
```

Argument Descriptions:    
`-c 'coords' for Source coordinates OR`    
`-n 'name' for Source name`    

      example accepted coordinate formats:    
      > RA,DEC or 'RA, DEC' in degrees    
      > '00h42m30s', '+41d12m00s' or 00h42m30s,+41d12m00s    
      > '00 42 30 +41 12 00'    
      > '00:42.5 +41:12'    
      if name:    
      > The name of the object to get coordinates for, e.g. 'M42'    

`-r 'radius' is the Integer search radius around the specified source location in arcmin.`    
      The cutouts will be of maximum width and height of 2*radius    

`-s 'surveys' is one or several surveys comma separated without spaces between.`       
```text
      Implemented surveys include:    
         - VLASS   
         - GLEAM    
            frequencies: f1 (072-103 MHz), f2 (103-034 MHz), f3 (139-170 MHz), f4 (170-231 MHz default)    
         - FIRST    
         - NVSS    
         - WISE    
            wavelengths: W1 (3.4μm default),  W2 (4.6μm),  W3 (12μm),  W4 (22μm)    
         - PANSTARRS    
            filters: g, r, i (default), z, y    
         - SDSS-I/II    
            filters: g (default), r, i    

        Filters/Frequencies/Wavelengths for each survey may be specified in the following formats:        
         > "WISE(w2),SDSS[g,r]"    
         > "WISE[w1],VLASS"    
         > "GLEAM(f1,f3)"    
         > WISE,VLASS    
```

`-o 'output' is the directory location to save output FITS images to.`    
      Output will be furthered separated into subfolders for the corresponding survey.    
      Default location is a folder named 'data_out/' in this current directory.    

`-g 'groupby' is an option to separate FITS results by "MOSAIC", "DATE-OBS", or "NONE" (default).`     

      > "MOSAIC": if the requested position and radius straddle boundaries in multiple      
                  FITS images for a given survey a mosaicked FITS file will be generated    
                  from all of these input images with each input image as an extension of    
                  the corresponding mosaicked FITS. Mosaics are largely provided for visual    
                  use only.    
      > "DATE-OBS": For surveys VLASS, FIRST, NVSS, or PanSTARRS a Mosaicked FITS is made    
                  (when needed) for every unique DATE-OBS.     
      > "NONE" (default): All resulting FITS images in the requested survey are returned    
                  without doing any mosaicking    

`-cf 'config' is to specify a YAML config file for settings, ex."config.yml".`    
      *Note: Specified command line args will overwrite these settings.`          

`-f "file" FOR FETCH_BATCH ONLY. The CSV file(s) name. `      

       CSV must at least have separate columns named "RA" and "Dec"    
       (or any of the variants below, but there can only be one variant of    
       RA and one of Dec per file). A column labelled "Name" or "NAME" may also be used.   
       For a given source, coordinates will be evaluated via "RA" and "Dec" if   
       they are non-empty. If a line does not have a valid coordinate position,   
       but does have a "Name" column value, the service will attempt to resolve   
       the source name.   
    
       Accepted variants of RA and Dec Column header names are:    
       R.A.   
       Right Ascension   
       RA (J2000)   
       R.A. (J2000)   
       Right Ascension (J2000)   
       RAJ2000   
       DEC   
       DEC.   
       Declination   
       DEC (J2000)   
       DEC. (J2000)   
       Declination (J2000)   
       DecJ2000   
    
       Source names will be resolved via the Sesame Name Resolver:    
       http://vizier.u-strasbg.fr/viz-bin/Sesame    

A sample command looks like:    
```bash 
python fetch_cutouts.py fetch -c 150,30 -s VLASS,WISE,FIRST,GLEAM -r 3 -g MOSAIC
```

<p align="right">(<a href="#top">back to top</a>)</p>



## Local Settings

Batch limit maximum is currently set to 1000; this is set as `self.MAX_BATCH_SIZE = 1000` in the file `cli_config.py` for the class `CLIConfig`. Please update for personal needs and system capability.    



## Output

The fetching script will output the `data_out` directory with the FITS files divided into directories according to the survey name. Success and/or failure results will be written onto `OUTlog.txt`

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Principal Investigator: [Professor Bryan Gaensler](mailto:bryan.gaensler@utoronto.ca)

Deputy Principal Investigator: [Professor Erik Rosolowsky](mailto:rosolowsky@ualberta.ca)

Project Manager: [Dr. Mathew Dionyssiou](mailto:mathew.dionyssiou@utoronto.ca)

Lead Software Developer: [Parthasarathy Venkataraman](mailto:p.venkataraman@utoronto.ca)



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* This Cutout Core repository based on common utilities and functionality forked from: http://orbit.dunlap.utoronto.ca/michelle.boyce/Continuum_common
* Applications currently using this common Core Cutout code include:
  - a Command Line Interface with instructions included above     
  - a public web service found at http://cutouts.cirada.ca  with project code hosted at https://gitlab.com/cirada/cutout_provider_gui

<p align="right">(<a href="#top">back to top</a>)</p>
