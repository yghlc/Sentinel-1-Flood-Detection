## ASF's searchAPI solution with python wrapper
import asf_search as asf
import os
import sys

## Set up output directory
region="Harvey2017"
parent_dir=["/data/Projects/GIFFT/" + region + "/sentinel1/"]
download_dir = ' '.join(parent_dir)

## Get login info
un = os.environ.get('ASF_API_USER')
pw = os.environ.get('ASF_API_PASS')

## Make output directory
if not os.path.exists(download_dir):
        os.makedirs(download_dir)


## ROI
wkt = 'POLYGON((-96.25 30.25, -96.25 29.25, -95.2 29.25, -95.2 30.25, -96.25 30.25))'
print('Searching... ... ...')
results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=wkt, start="2017-08-10", end="2017-09-10",
beamMode="IW", processingLevel="GRD_HD")
print('Found %s results' % (len(results)))
session = asf.ASFSession()
session.auth_with_creds(un, pw)
print('Downloading... ... ...')
results.download(path=download_dir, session=session) # download results to a path
print('Finished Download')
#
#

## Save results to an output log
log_filename=(download_dir + region + "_download_log.txt")
print(' ')
print('Saving results to ', log_filename)
stdoutOrigin=sys.stdout
# sys.stdout = open (download_dir + region + "_download_log.txt", "w")
sys.stdout = open (log_filename, "w")
print(results)
sys.stdout.close()
sys.stdout = stdoutOrigin
