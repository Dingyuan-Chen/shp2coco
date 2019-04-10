# shp2coco
shp2coco is a a tool to help create COCO datasets from .shp file (ArcGIS format). 
It includes functions:
  1:mask tif with shape file
  2:crop tif and mask
  2:generate annotations in uncompressed RLE ("crowd") and polygons in the format COCO requires.

usage:
python shape_to_coco.py
