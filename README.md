# Sentinel-1-Flood-Detection


### TODO
- [x] change the binary map from Float to Byte, add compression, reducing size of each Geotiff from ~3 GB to ~10 MB.
- [x] change python script to command 
- [x] add to option: not outputting png files (histogram)
- [x] add color table to the binary map 
- [x] add a temporal folder, for tmp file, reducing the IO burden for network disks
- [x] reduce the CPU memory cost; running flooding detection in parallel
- [x] For BCV: removing extreme values; require at least 10000 pixels
- [x] local minimum: checking balance between two classes: one class presents at least 20% 
- [ ] Use "a Hierarchical Split-Based Approach (Chini et al. 2017)" for tiling 
