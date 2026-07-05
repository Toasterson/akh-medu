# Create a ZFS pool and dataset with specific properties

A 200MB file at `/var/tmp/akh-e2e-vdev.img` has been prepared as a vdev.
Using it:

1. Create a zpool named `akhtank` backed by that file.
2. Create a dataset `akhtank/data` with:
   - `compression` set to `lz4`
   - `quota` set to `50M`
   - `mountpoint` set to `/akh-data`

Report back once the pool is healthy and the dataset is configured as
above.
