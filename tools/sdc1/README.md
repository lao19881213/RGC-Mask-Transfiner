# Data Sets Create Steps  
## create train image  
Cutout a single training image that contains all the objects on the training catalog from the "field_fits"
```
srun -N 1 -p hw -w hw-x86-cpu11 python create_train_data_B1.py 
```
## split train fits image  
```
sbatch -N 1 -p all-x86-cpu -w hw-x86-cpu01 split_fits.sh
```
## fits to png
```
./fits2png_batch.sh
```
## add pgsphere to PostgreSQL
```
git clone https://github.com/akorotkov/pgsphere
ssh purley-x86-cpu01
gmake USE_PGXS=1 PG_CONFIG=/usr/bin/pg_config
gmake USE_PGXS=1 PG_CONFIG=/usr/bin/pg_config install
su postgres
createdb blao -U postgres
psql -c "CREATE EXTENSION pg_sphere;" blao
psql -U postgres sd1 < pg_sphere--1.0.sql
vim /var/lib/pgsql/data/pg_hba.conf
change to
local   all     all             trust
service postgresql restart
```
## create xml file
```
sbatch -N 1 -p purley-cpu -w purley-x86-cpu01 create_xml_anno.sh
``` 
