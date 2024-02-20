#!/bin/bash  

export PATH=/home/data0/lbq/software/miriad/linux64/bin:$PATH

IMG_DIR="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid"

#bmaj_fnl="9.0" #arcsec 0.0025 deg `echo "$bmin*3600.0" | bc -l`
#bmin_fnl="7.6" #arcsec 0.00211111111111111 deg
#bpa_fnl="71.5" #deg 
for id in {4..4};
do
IMG_FILES="/home/data0/lbq/inference_data/ASKAP_RACS/RACS_mid/part${id}_fits_re.txt"

for fn in `cat $IMG_FILES`;
do
    if [ ! -d "${IMG_DIR}/part${id}_conv" ];then
       mkdir -p ${IMG_DIR}/part${id}_conv
    else
       echo "${IMG_DIR}/part${id}_conv is already exists"
    fi
    echo "Processing ${fn} ... ..."
  
    if [ ! -f "${IMG_DIR}/part${id}_conv/${fn%%.fits}_fixed.fits" ]; then 
       cd ${IMG_DIR}/part${id}_conv  
       ln -s ${IMG_DIR}/part${id}/${fn} ${fn}
       image=${fn}
       bmaj=$(fitsheader $image | grep BMAJ | awk 'NR==1{print $3}')
       #bmaj=`echo "$bmaj" | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}'`
       bmaj=`echo "$bmaj*3600.0" | bc -l` # convert from deg to arcsec
       bmaj=`echo $bmaj | awk '{print ($0-int($0)<0.000001)?int($0):int($0)+1}'` # int
       bmaj=$(($bmaj+1))
       bmin=`echo "scale=2; $bmaj/1.125" | bc`
       bmin=`echo $bmin | awk '{print ($0-int($0)<0.000001)?int($0):int($0)+1}'`
       
       bmaj_fnl=$bmaj
       bmin_fnl=$bmin
       bpa=$(fitsheader $image | grep BPA | awk '{print $3}') 
       bpa=`echo $bpa | awk '{print ($0-int($0)<0.000001)?int($0):int($0)+1}'` #
       bpa_fnl=$bpa
       echo $bmaj_fnl,$bmin_fnl, $bpa_fnl
       echo "Convolving image $image"
       fits op=xyin in=$image out=map
       puthd in=map/bunit value="JY/BEAM"
       convol map=map fwhm=$bmaj_fnl,$bmin_fnl pa=$bpa_fnl out=map2 options=final
       #convol map=map fwhm=9,8 pa=72 out=map2 options=final
       fits op=xyout in=map2 out=${image%%.fits}_fixed.fits
       rm -rf map map2
       unlink $image
     else
       echo "${IMG_DIR}/part${id}_conv/${fn%%.fits}_fixed.fits already exists!"
     fi
done
done

