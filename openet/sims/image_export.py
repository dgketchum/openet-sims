import os
import sys
import time
from datetime import datetime

import ee
import geopandas as gpd
from tqdm import tqdm

from openet.sims.ee_utils import is_authorized, get_lanid
import openet


sys.path.insert(0, os.path.abspath('../..'))
sys.setrecursionlimit(5000)


# Landsat collections supported by SIMS in this repo
LANDSAT_COLLECTIONS = [
    'LANDSAT/LT04/C02/T1_L2',
    'LANDSAT/LT05/C02/T1_L2',
    'LANDSAT/LE07/C02/T1_L2',
    'LANDSAT/LC08/C02/T1_L2',
    'LANDSAT/LC09/C02/T1_L2',
]

# Irrigation mask sources and regions (matching PT-JPL workflow)
IRR = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
STATES = ['AZ', 'CA', 'CO', 'ID', 'MT', 'NM', 'NV', 'OR', 'UT', 'WA', 'WY']
WEST_STATES = 'users/dgketchum/boundaries/western_11_union'
EAST_STATES = 'users/dgketchum/boundaries/eastern_38_dissolved'


def export_sims_zonal_stats(
    shapefile,
    bucket,
    feature_id='FID',
    select=None,
    start_yr=2000,
    end_yr=2024,
    mask_type='irr',
    check_dir=None,
    state_col='state',
    buffer=False,
):
    """Export per-scene SIMS ET fraction zonal means for polygons to GCS CSVs.

    Mirrors the PT-JPL zonal stats workflow but computes SIMS ET fraction.

    Parameters
    ----------
    shapefile : str
        Path to polygon shapefile with feature IDs.
    bucket : str
        GCS bucket name (no scheme). Tables saved under sims_tables/<fid>/.
    feature_id : str, optional
        Field name for feature identifier.
    select : list, optional
        Optional list of feature IDs to process.
    start_yr, end_yr : int
        Inclusive year range to process.
    mask_type : {'irr','inv_irr','no_mask'}
        Irrigation masking strategy.
    check_dir : str or None
        If set, skip exports when CSV already exists at check_dir/<fid>/<desc>.csv.
    state_col : str
        Column with state abbreviation to choose mask source.
    buffer : float or bool
        If truthy, buffer geometries by this distance before use.
    """
    df = gpd.read_file(shapefile)
    df = df.set_index(feature_id, drop=False)

    if buffer:
        df.geometry = df.geometry.buffer(buffer)

    original_crs = df.crs
    if original_crs and original_crs.srs != 'EPSG:4326':
        df = df.to_crs(4326)

    irr_coll = ee.ImageCollection(IRR)
    s, e = '1987-01-01', '2024-12-31'
    # Build a minimum-year mask to constrain annual masks
    remap = irr_coll.filterDate(s, e).select('classification').map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(5)
    east = ee.FeatureCollection(EAST_STATES)
    lanid = get_lanid()

    for fid, row in tqdm(df.iterrows(), desc='Export SIMS zonal stats', total=df.shape[0]):
        if row['geometry'].geom_type == 'Point':
            # Points are not supported in this workflow
            continue
        elif row['geometry'].geom_type == 'Polygon':
            polygon = ee.Geometry(row.geometry.__geo_interface__)
        else:
            # Multi* or other geometry types are skipped
            continue

        if select is not None and fid not in select:
            continue

        for year in range(start_yr, end_yr + 1):

            desc = f'sims_etf_{fid}_{mask_type}_{year}'
            fn_prefix = os.path.join('sims_tables', mask_type, str(fid), desc)

            if check_dir:
                f = os.path.join(check_dir, mask_type, str(fid), f'{desc}.csv')
                if os.path.exists(f):
                    print(f'{f} exists, skipping')
                    continue

            if mask_type in ['irr', 'inv_irr']:
                if state_col in row and row[state_col] in STATES:
                    irr = (
                        irr_coll
                        .filterDate(f'{year}-01-01', f'{year}-12-31')
                        .select('classification').mosaic()
                    )
                    irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
                else:
                    irr_mask = lanid.select(f'irr_{year}').clip(east)
                    irr = ee.Image(1).subtract(irr_mask)
            else:
                irr, irr_mask = None, None

            coll = openet.sims.Collection(
                LANDSAT_COLLECTIONS,
                start_date=f'{year}-01-01',
                end_date=f'{year}-12-31',
                geometry=polygon,
                cloud_cover_max=70,
            )

            scenes = coll.get_image_ids()
            scenes = list(set(scenes))
            scenes = sorted(scenes, key=lambda item: item.split('_')[-1])

            first, bands = True, None
            selectors = [feature_id]

            for img_id in scenes:
                splt = img_id.split('/')[-1].split('_')
                _name = '_'.join(splt[-3:])
                selectors.append(_name)

                sims_img = openet.sims.Image.from_landsat_c2_sr(img_id)
                etf_img = sims_img.et_fraction.rename(_name)

                if mask_type == 'no_mask':
                    etf_img = etf_img.clip(polygon)
                elif mask_type == 'irr':
                    etf_img = etf_img.clip(polygon).mask(irr_mask)
                elif mask_type == 'inv_irr':
                    etf_img = etf_img.clip(polygon).mask(irr.gt(0))

                if first:
                    bands = etf_img
                    first = False
                else:
                    bands = bands.addBands([etf_img])

            if bands is None:
                continue

            fc = ee.FeatureCollection(ee.Feature(polygon, {feature_id: fid}))
            data = bands.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=30)

            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=desc,
                bucket=bucket,
                fileNamePrefix=fn_prefix,
                fileFormat='CSV',
                selectors=selectors,
            )

            try:
                task.start()
                print(desc)
            except ee.ee_exception.EEException as e:
                error_message = str(e)
                if 'payload size exceeds the limit' in error_message:
                    print(f'Payload size limit exceeded for {desc}. Skipping task.')
                    continue
                elif 'many tasks already in the queue' in error_message:
                    print(f'Task queue full. Waiting 10 minutes to retry {desc}...')
                    time.sleep(600)
                    task.start()
                else:
                    raise


if __name__ == '__main__':
    # Ensure Earth Engine auth is valid (delegates to shared utils)
    is_authorized()

    project = '5_Flux_Ensemble'
    root = '/data/ssd2/swim'
    data = os.path.join(root, project, 'data')
    project_ws_ = os.path.join(root, project)
    if not os.path.isdir(root):
        root = '/home/dgketchum/PycharmProjects/swim-rs'
        project_ws_ = os.path.join(root, 'tutorials', project)
        data = os.path.join(project_ws_, 'data')

    shapefile_ = os.path.join(data, 'gis', 'flux_footprints_3p.shp')
    chk_dir = os.path.join(data, 'landsat', 'extracts', 'sims_etf')

    FEATURE_ID = 'site_id'

    export_sims_zonal_stats(
        shapefile=shapefile_,
        bucket='wudr',
        feature_id=FEATURE_ID,
        start_yr=1987,
        end_yr=2024,
        select=None,
        mask_type='inv_irr',
        check_dir=chk_dir,
        state_col='state',
        buffer=None,
    )

    # ========================= EOF ==========================================

