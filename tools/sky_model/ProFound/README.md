# ProFound (R package)

## Installation

### Getting R

#### Linux

Centos or Fedora:	`sudo yum install R`

Ubuntu:	`sudo apt-get install r-base-dev`

### Getting ProFound

```R
install.packages('devtools')
devtools::install_github("asgr/ProFound")
library(ProFound)
```

#### Package Dependencies

```R
install.packages(c('magicaxis', 'FITSio', 'data.table')) # Required packages
install.packages(c('knitr', 'rmarkdown', 'EBImage', 'akima', 'imager', 'LaplacesDemon')) # Suggested packages
install.packages('remotes')
remotes::install_github("asgr/ProFound")
```

```
git clone https://github.com/asgr/ProFound.git
tar zcvf ProFound.tar.gz ProFound
R CMD INSTALL ProFound.tar.gz
```
