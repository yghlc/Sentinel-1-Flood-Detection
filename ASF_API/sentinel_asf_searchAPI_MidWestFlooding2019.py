## ASF's searchAPI solution with python wrapper
import asf_search as asf
import os
import sys

## Set up output directory
region="Midwest2019"
parent_dir=["/data/Projects/GIFFT/" + region + "/sentinel1/"]
download_dir = ' '.join(parent_dir)

## Get login info
un = os.environ.get('ASF_API_USER')
pw = os.environ.get('ASF_API_PASS')
# print('Earth Data username: ', un)
# print('Earth Data password: ', pw)


## Make output directory
if not os.path.exists(download_dir):
        os.makedirs(download_dir)


## ROI
wkt = 'POLYGON((-98.23974609375 43.34116005412307, -98.23974609375 38.85682013474361, -92.94433593749 38.85682013474361, -92.94433593749 43.34116005412307, -98.23974609375 43.34116005412307))'
print('Searching... ... ...')
results = asf.geo_search(platform=[asf.PLATFORM.SENTINEL1], intersectsWith=wkt, start="2019-04-01", end="2019-04-30",
beamMode="IW", processingLevel="GRD_HD")
print('Found %s results' % (len(results)))
session = asf.ASFSession()
session.auth_with_creds(un, pw)
print('Downloading... ... ...')
results.download(path=download_dir, session=session) # download results to a path
print('Finished Download')



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
