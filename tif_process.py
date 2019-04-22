# Date:2019.04.10
# Author: DuncanChen
# A tool implementation on gdal and geotool API
# functions:
# 1. get mask raster with shapefile
# 2. clip raster and shapefile with grid

from PIL import Image, ImageDraw
import os
from osgeo import gdal, gdalnumeric
import numpy as np
import ogr
import glob
gdal.UseExceptions()


class GeoTiff(object):
    def __init__(self, tif_path):
        """
        A tool for Remote Sensing Image
        Args:
            tif_path: tif path
        Examples::
            >>> tif = GeoTif('xx.tif')
            # if you want to clip tif with grid reserved geo reference
            >>> tif.clip_tif_with_grid(512, 'out_dir')
            # if you want to clip tif with shape file
            >>> tif.clip_tif_with_shapefile('shapefile.shp', 'save_path.tif')
            # if you want to mask tif with shape file
            >>> tif.mask_tif_with_shapefile('shapefile.shp', 'save_path.tif')
        """
        self.dataset = gdal.Open(tif_path)
        self.bands_count = self.dataset.RasterCount
        # get each band
        self.bands = [self.dataset.GetRasterBand(i + 1) for i in range(self.bands_count)]
        self.col = self.dataset.RasterXSize
        self.row = self.dataset.RasterYSize
        self.geotransform = self.dataset.GetGeoTransform()
        self.src_path = tif_path
        self.mask = None
        self.mark = None

    def get_left_top(self):
        return self.geotransform[3], self.geotransform[0]

    def get_pixel_height_width(self):
        return abs(self.geotransform[5]), abs(self.geotransform[1])

    def __getitem__(self, *args):
        """

        Args:
            *args: range, an instance of tuple, ((start, stop, step), (start, stop, step))

        Returns:
            res: image block , array ,[bands......, height, weight]

        """
        if isinstance(args[0], tuple) and len(args[0]) == 2:
            # get params
            start_row, end_row = args[0][0].start, args[0][0].stop
            start_col, end_col = args[0][1].start, args[0][1].stop
            start_row = 0 if start_row is None else start_row
            start_col = 0 if start_col is None else start_col
            num_row = self.row if end_row is None else (end_row - start_row)
            num_col = self.col if end_col is None else (end_col - start_col)
            # dataset read image array
            res = self.dataset.ReadAsArray(start_col, start_row, num_col, num_row)
            return res
        else:
            raise NotImplementedError('the param should be [a: b, c: d] !')

    def clip_tif_with_grid(self, clip_size, begin_id, out_dir):
        """
        clip image with grid
        Args:
            clip_size: int
            out_dir: str

        Returns:

        """
        if not os.path.exists(out_dir):
            # check the dir
            os.makedirs(out_dir)
            print('create dir', out_dir)

        row_num = int(self.row / clip_size)
        col_num = int(self.col / clip_size)

        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        count = 1
        for i in range(row_num):
            for j in range(col_num):
                # if begin_id+i*col_num+j in self.mark:
                #     continue
                clipped_image = np.array(self[i * clip_size: (i + 1) * clip_size, j * clip_size: (j + 1) * clip_size])
                clipped_image = clipped_image.astype(np.int8)

                try:
                    save_path = os.path.join(out_dir, '%d.tif' % (begin_id+i*col_num+j))
                    save_image_with_georef(clipped_image, gtiffDriver,
                                           self.dataset, j*clip_size, i*clip_size, save_path)
                    print('clip successfully！(%d/%d)' % (count, row_num * col_num))
                    count += 1
                except Exception:
                    raise IOError('clip failed!%d' % count)

        return row_num * col_num

    def clip_mask_with_grid(self, clip_size, begin_id, out_dir):
        """
        clip mask with grid
        Args:
            clip_size: int
            out_dir: str

        Returns:

        """
        if not os.path.exists(out_dir):
            # check the dir
            os.makedirs(out_dir)
            print('create dir', out_dir)

        row_num = int(self.row / clip_size)
        col_num = int(self.col / clip_size)

        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        # self.mark = []

        count = 1
        for i in range(row_num):
            for j in range(col_num):
                clipped_image = np.array(self.mask[0, i * clip_size: (i + 1) * clip_size, j * clip_size: (j + 1) * clip_size])
                ins_list = np.unique(clipped_image)
                # if len(ins_list) <= 1:
                #     self.mark.append(begin_id+i*col_num+j)
                #     continue
                ins_list = ins_list[1:]
                for id in range(len(ins_list)):
                    bg_img = np.zeros((clipped_image.shape)).astype(np.int8)
                    if ins_list[id] > 0:
                        bg_img[np.where(clipped_image == ins_list[id])] = 255
                    try:
                        save_path = os.path.join(out_dir, '%d_%s_%d.tif' % (begin_id+i*col_num+j, 'greenhouse', id))
                        save_image_with_georef(bg_img, gtiffDriver,
                                               self.dataset, j*clip_size, i*clip_size, save_path)
                        print('clip mask successfully！(%d/%d)' % (count, row_num * col_num))
                        count += 1
                    except Exception:
                        raise IOError('clip failed!%d' % count)

    def world2Pixel(self, x, y):
        """
        Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
        the pixel location of a geospatial coordinate
        """
        ulY, ulX = self.get_left_top()
        distY, distX = self.get_pixel_height_width()

        pixel_x = abs(int((x - ulX) / distX))
        pixel_y = abs(int((ulY - y) / distY))
        pixel_y = self.row if pixel_y > self.row else pixel_y
        pixel_x = self.col if pixel_x > self.col else pixel_x
        return pixel_x, pixel_y

    def mask_tif_with_shapefile(self, shapefile_path, label=255):
        """
        mask tif with shape file, supported point, line, polygon and multi polygons
        Args:
            shapefile_path:
            save_path:
            label:

        Returns:

        """
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(shapefile_path, 0)
        if dataSource is None:
            raise IOError('could not open!')
        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        layer = dataSource.GetLayer(0)
        # # Convert the layer extent to image pixel coordinates
        minX, maxX, minY, maxY = layer.GetExtent()
        ulX, ulY = self.world2Pixel(minX, maxY)

        # initialize mask drawing
        rasterPoly = Image.new("I", (self.col, self.row), 0)
        rasterize = ImageDraw.Draw(rasterPoly)

        feature_num = layer.GetFeatureCount()  # get poly count
        for i in range(feature_num):
            points = []  # store points
            pixels = []  # store pixels
            feature = layer.GetFeature(i)
            geom = feature.GetGeometryRef()
            feature_type = geom.GetGeometryName()

            if feature_type == 'POLYGON' or 'MULTIPOLYGON':
                # multi polygon operation
                # 1. use label to mask the max polygon
                # 2. use -label to mask the other polygon
                for j in range(geom.GetGeometryCount()):
                    sub_polygon = geom.GetGeometryRef(j)
                    if feature_type == 'MULTIPOLYGON':
                        sub_polygon = sub_polygon.GetGeometryRef(0)
                    for p_i in range(sub_polygon.GetPointCount()):
                        px = sub_polygon.GetX(p_i)
                        py = sub_polygon.GetY(p_i)
                        points.append((px, py))

                    for p in points:
                        origin_pixel_x, origin_pixel_y = self.world2Pixel(p[0], p[1])
                        # the pixel in new image
                        new_pixel_x, new_pixel_y = origin_pixel_x, origin_pixel_y
                        pixels.append((new_pixel_x, new_pixel_y))

                    rasterize.polygon(pixels, i+1)
                    pixels = []
                    points = []
                    if feature_type != 'MULTIPOLYGON':
                        label = -abs(label)

                # restore the label value
                label = abs(label)
            else:
                for j in range(geom.GetPointCount()):
                    px = geom.GetX(j)
                    py = geom.GetY(j)
                    points.append((px, py))

                for p in points:
                    origin_pixel_x, origin_pixel_y = self.world2Pixel(p[0], p[1])
                    # the pixel in new image
                    new_pixel_x, new_pixel_y = origin_pixel_x, origin_pixel_y
                    pixels.append((new_pixel_x, new_pixel_y))

                feature.Destroy()  # delete feature

                if feature_type == 'LINESTRING':
                    rasterize.line(pixels, i+1)
                if feature_type == 'POINT':
                    # pixel x, y
                    rasterize.point(pixels, i+1)

        mask = np.array(rasterPoly)
        self.mask = mask[np.newaxis, :]  # extend an axis to three

    def clip_tif_and_shapefile(self, clip_size, begin_id, shapefile_path, out_dir):
        self.mask_tif_with_shapefile(shapefile_path)
        self.clip_mask_with_grid(clip_size=clip_size, begin_id=begin_id, out_dir=out_dir + '/annotations')
        pic_id = self.clip_tif_with_grid(clip_size=clip_size, begin_id=begin_id, out_dir=out_dir + '/greenhouse_2019')
        return pic_id

def channel_first_to_last(image):
    """

    Args:
        image: 3-D numpy array of shape [channel, width, height]

    Returns:
        new_image: 3-D numpy array of shape [height, width, channel]
    """
    new_image = np.transpose(image, axes=[1, 2, 0])
    return new_image

def channel_last_to_first(image):
    """

    Args:
        image: 3-D numpy array of shape [channel, width, height]

    Returns:
        new_image: 3-D numpy array of shape [height, width, channel]
    """
    new_image = np.transpose(image, axes=[2, 0, 1])
    return new_image

def save_image_with_georef(image, driver, original_ds, offset_x=0, offset_y=0, save_path=None):
    """

    Args:
        save_path: str, image save path
        driver: gdal IO driver
        image: an instance of ndarray
        original_ds: a instance of data set
        offset_x: x location in data set
        offset_y: y location in data set

    Returns:

    """
    # get Geo Reference
    ds = gdalnumeric.OpenArray(image)
    gdalnumeric.CopyDatasetInfo(original_ds, ds, xoff=offset_x, yoff=offset_y)
    driver.CreateCopy(save_path, ds)
    # write by band
    clip = image.astype(np.int8)
    # write the dataset
    if len(image.shape)==3:
        for i in range(image.shape[0]):
            ds.GetRasterBand(i + 1).WriteArray(clip[i])
    else:
        ds.GetRasterBand(1).WriteArray(clip)
    del ds

def define_ref_predict(tif_dir, mask_dir, save_dir):
    """
    define reference for raster referred to a geometric raster.
    Args:
        tif_dir: the dir to save referenced raster
        mask_dir:
        save_dir:

    Returns:

    """
    tif_list = glob.glob(os.path.join(tif_dir, '*.tif'))

    mask_list = glob.glob(os.path.join(mask_dir, '*.png'))
    mask_list += (glob.glob(os.path.join(mask_dir, '*.jpg')))
    mask_list += (glob.glob(os.path.join(mask_dir, '*.tif')))

    tif_list.sort()
    mask_list.sort()

    os.makedirs(save_dir, exist_ok=True)
    gtiffDriver = gdal.GetDriverByName('GTiff')
    if gtiffDriver is None:
        raise ValueError("Can't find GeoTiff Driver")
    for i in range(len(tif_list)):
        save_name = tif_list[i].split('\\')[-1]
        save_path = os.path.join(save_dir, save_name)
        tif = GeoTiff(tif_list[i])
        mask = np.array(Image.open(mask_list[i]))
        mask = channel_last_to_first(mask)
        save_image_with_georef(mask, gtiffDriver, tif.dataset, save_path=save_path)

class GeoShaplefile(object):
    def __init__(self, file_path=""):
        self.file_path = file_path
        self.layer = ""
        self.minX, self.maxX, self.minY, self.maxY = (0, 0, 0, 0)
        self.feature_type = ""
        self.feature_num = 0
        self.open_shapefile()
    def open_shapefile(self):
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataSource = driver.Open(self.file_path, 0)
        if dataSource is None:
            raise IOError('could not open!')
        gtiffDriver = gdal.GetDriverByName('GTiff')
        if gtiffDriver is None:
            raise ValueError("Can't find GeoTiff Driver")

        self.layer = dataSource.GetLayer(0)
        self.minX, self.maxX, self.minY, self.maxY = self.layer.GetExtent()
        self.feature_num = self.layer.GetFeatureCount()  # get poly count
        if self.feature_num > 0:
            polygon = self.layer.GetFeature(0)
            geom = polygon.GetGeometryRef()
            # feature type
            self.feature_type = geom.GetGeometryName()

def clip_from_file(clip_size, root, img_path, shp_path):
    img_list = os.listdir(root + '/' + img_path)
    n_img = len(img_list)
    pic_id = 0
    for i in range(n_img):
        tif = GeoTiff(root + '/' + img_path + '/' + img_list[i])
        img_id = img_list[i].split('.', 1)[0]
        pic_num = tif.clip_tif_and_shapefile(clip_size, pic_id, root + '/' + shp_path + '/' + img_id + '/' + img_id + '.shp', root + '/dataset')
        pic_id += pic_num

if __name__ == '__main__':
    root = r'./example_data/original_data'
    img_path = 'img'
    shp_path = 'shp'
    clip_from_file(512, root, img_path, shp_path)
