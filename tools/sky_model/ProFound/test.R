library(ProFound)
imagefile="J001636+120652.fits"
image=readFITS(imagefile)
image_blind=profoundProFound(image, plot=TRUE, skycut=3.5, rotstats=TRUE, boundstats=TRUE, nearstats=TRUE,groupstats=TRUE, groupby='segim', verbose=TRUE)
write.csv(image_blind$groupstats,file='file_name.csv', quote=FALSE, row.names=FALSE)
segim_model=image_blind$group$groupim
segim_model[image_blind$group$groupim!=0]=1
segim_model[image_blind$group$groupim==0]=as.numeric(NaN)
model=(image$imDat-image_blind$sky)*segim_model

write.csv(model, file="model.csv", quote=FALSE, row.names=FALSE)

write.table(model, file="model.txt")
