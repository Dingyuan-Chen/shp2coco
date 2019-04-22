# shp2coco
shp2coco is a tool to help create `COCO` datasets from `.shp` file (ArcGIS format). <br>

It includes:<br>
1:mask tif with shape file.<br>
2:crop tif and mask.<br>
3:slice the dataset into training, eval and test subset.<br>
4:generate annotations in uncompressed RLE ("crowd") and polygons in the format COCO requires.<br>

This project is based on [geotool](https://github.com/Kindron/geotool) and [pycococreator](https://github.com/waspinator/pycococreator)

## Usage:
If you need to generate annotations in the COCO format, try the following:<br>
`python shape_to_coco.py`<br>
If you need to visualize annotations, try the following:<br>
`python visualize_coco.py`<br>

## Example:
![example](https://github.com/DuncanChen2018/shp2coco/blob/master/example_data/example.png)

## Thanks to the Third Party Libs
[geotool](https://github.com/Kindron/geotool)<br>
[pycococreator](https://github.com/waspinator/pycococreator)<br>
