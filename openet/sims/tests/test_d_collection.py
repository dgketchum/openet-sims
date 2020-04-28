import pprint

import ee
import pytest

import openet.sims as sims
import openet.sims.utils as utils
# TODO: import utils from openet.core
# import openet.core.utils as utils


COLLECTIONS = ['LANDSAT/LC08/C01/T1_SR', 'LANDSAT/LE07/C01/T1_SR']
SCENE_ID_LIST = sorted(['LC08_044033_20170716', 'LE07_044033_20170708',
                        'LE07_044033_20170724'])
START_DATE = '2017-07-01'
END_DATE = '2017-08-01'
SCENE_GEOM = (-121.91, 38.99, -121.89, 39.01)
SCENE_POINT = (-121.9, 39)
VARIABLES = {'et', 'et_fraction', 'et_reference'}
TEST_POINT = (-121.5265, 38.7399)

default_coll_args = {
    'collections': COLLECTIONS,
    'geometry': ee.Geometry.Point(SCENE_POINT),
    'start_date': START_DATE,
    'end_date': END_DATE,
    'variables': list(VARIABLES),
    'cloud_cover_max': 70,
    'et_reference_source': 'IDAHO_EPSCOR/GRIDMET',
    'et_reference_band': 'eto',
    'et_reference_factor': 0.85,
    'et_reference_resample': 'nearest',
    'model_args': {},
    'filter_args': {},
}

def default_coll_obj(**kwargs):
    args = default_coll_args.copy()
    args.update(kwargs)
    return sims.Collection(**args)


def parse_scene_id(output_info):
    output = [x['properties']['system:index'] for x in output_info['features']]
    # Strip merge indices (this works for Landsat and Sentinel image IDs
    return sorted(['_'.join(x.split('_')[-3:]) for x in output])


def test_Collection_init_default_parameters():
    """Test if init sets default parameters"""
    args = default_coll_args.copy()
    # These values are being set above but have defaults that need to be checked
    del args['et_reference_source']
    del args['et_reference_band']
    del args['et_reference_factor']
    del args['et_reference_resample']
    del args['variables']

    m = sims.Collection(**args)
    assert m.variables == None
    assert m.et_reference_source == None
    assert m.et_reference_band == None
    assert m.et_reference_factor == None
    assert m.et_reference_resample == None
    assert m.cloud_cover_max == 70
    assert m.model_args == {}
    assert m.filter_args == {}
    assert set(m._interp_vars) == {'ndvi', 'et_fraction'}


def test_Collection_init_collection_str(coll_id='LANDSAT/LC08/C01/T1_SR'):
    """Test if a single coll_id str is converted to a single item list"""
    assert default_coll_obj(collections=coll_id).collections == [coll_id]


def test_Collection_init_cloud_cover_max_str():
    """Test if cloud_cover_max strings are converted to float"""
    assert default_coll_obj(cloud_cover_max='70').cloud_cover_max == 70


@pytest.mark.parametrize(
    'coll_id, start_date, end_date',
    [
        ['LANDSAT/LT04/C01/T1_SR', '1981-01-01', '1982-01-01'],
        ['LANDSAT/LT04/C01/T1_SR', '1994-01-01', '1995-01-01'],
        ['LANDSAT/LT05/C01/T1_SR', '1983-01-01', '1984-01-01'],
        ['LANDSAT/LT05/C01/T1_SR', '2012-01-01', '2013-01-01'],
        ['LANDSAT/LE07/C01/T1_SR', '1998-01-01', '1999-01-01'],
        ['LANDSAT/LC08/C01/T1_SR', '2012-01-01', '2013-01-01'],
    ]
)
def test_Collection_init_collection_filter(coll_id, start_date, end_date):
    """Test that collection IDs are filtered based on start/end dates"""
    # The target collection ID should be removed from the collections lists
    assert default_coll_obj(collections=coll_id, start_date=start_date,
                            end_date=end_date).collections == []


def test_Collection_init_startdate_exception():
    """Test if Exception is raised for invalid start date formats"""
    with pytest.raises(ValueError):
        default_coll_obj(start_date='1/1/2000', end_date='2000-01-02')


def test_Collection_init_enddate_exception():
    """Test if Exception is raised for invalid end date formats"""
    with pytest.raises(ValueError):
        default_coll_obj(start_date='2000-01-01', end_date='1/2/2000')


def test_Collection_init_swapped_date_exception():
    """Test if Exception is raised when start_date == end_date"""
    with pytest.raises(ValueError):
        default_coll_obj(start_date='2017-01-01', end_date='2017-01-01')


def test_Collection_init_invalid_collections_exception():
    """Test if Exception is raised for an invalid collection ID"""
    with pytest.raises(ValueError):
        default_coll_obj(collections=['FOO'])


# CGM - This test is not needed since only Landsat SR collections are supported
# def test_Collection_init_duplicate_collections_exception():
#     """Test if Exception is raised for duplicate Landsat types"""
#     with pytest.raises(ValueError):
#         default_coll_obj(collections=['LANDSAT/LC08/C01/T1_RT_TOA',
#                                       'LANDSAT/LC08/C01/T1_TOA'])
#     with pytest.raises(ValueError):
#         default_coll_obj(collections=['LANDSAT/LC08/C01/T1_SR',
#                                       'LANDSAT/LC08/C01/T1_TOA'])


def test_Collection_init_cloud_cover_exception():
    """Test if Exception is raised for an invalid cloud_cover_max"""
    with pytest.raises(TypeError):
        default_coll_obj(cloud_cover_max='A')
    with pytest.raises(ValueError):
        default_coll_obj(cloud_cover_max=-1)
    with pytest.raises(ValueError):
        default_coll_obj(cloud_cover_max=101)


def test_Collection_build_default():
    output = utils.getinfo(default_coll_obj()._build())
    assert output['type'] == 'ImageCollection'
    assert parse_scene_id(output) == SCENE_ID_LIST
    # For the default build, check that the target variables are returned also
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


def test_Collection_build_variables():
    output = utils.getinfo(default_coll_obj()._build(variables=['ndvi']))
    assert {y['id'] for x in output['features'] for y in x['bands']} == {'ndvi'}


def test_Collection_build_dates():
    """Check that dates passed to build function override Class dates"""
    coll_obj = default_coll_obj(start_date='2017-08-01', end_date='2017-09-01')
    output = utils.getinfo(coll_obj._build(
        start_date='2017-07-16', end_date='2017-07-17'))
    assert parse_scene_id(output) == ['LC08_044033_20170716']


def test_Collection_build_landsat_toa():
    """Test if the Landsat TOA (non RT) collections can be built"""
    coll_obj = default_coll_obj(
        collections=['LANDSAT/LC08/C01/T1_TOA', 'LANDSAT/LE07/C01/T1_TOA'])
    output = utils.getinfo(coll_obj._build())
    assert parse_scene_id(output) == SCENE_ID_LIST
    assert VARIABLES == {y['id'] for x in output['features'] for y in x['bands']}


def test_Collection_build_landsat_sr():
    """Test if the Landsat SR collections can be built"""
    coll_obj = default_coll_obj(
        collections=['LANDSAT/LC08/C01/T1_SR', 'LANDSAT/LE07/C01/T1_SR'])
    output = utils.getinfo(coll_obj._build())
    assert parse_scene_id(output) == SCENE_ID_LIST
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


# CGM - Non Landsat SR collections not currently supported
# def test_Collection_build_sentinel_toa():
#     """Test if the Sentinel 2 TOA collections can be built"""
#     coll_obj = default_coll_obj(
#         collections=['COPERNICUS/S2'], start_date='2017-05-01',
#         end_date='2017-05-15')
#     output = utils.getinfo(coll_obj._build())
#     expected = ['20170510T184921_20170510T185915_T10SEJ']
#     assert parse_scene_id(output) == expected
#     assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


def test_Collection_build_exclusive_enddate():
    """Test if the end_date is exclusive"""
    output = utils.getinfo(default_coll_obj(end_date='2017-07-24')._build())
    assert [x for x in parse_scene_id(output) if int(x[-8:]) >= 20170724] == []


def test_Collection_build_cloud_cover():
    """Test if the cloud cover max parameter is being applied"""
    # CGM - The filtered images should probably be looked up programmatically
    output = utils.getinfo(default_coll_obj(cloud_cover_max=0.5)._build(
        variables=['et']))
    assert 'LE07_044033_20170724' not in parse_scene_id(output)


def test_Collection_build_filter_args():
    # Need to test with two collections to catch bug when deepcopy isn't used
    collections = ['LANDSAT/LC08/C01/T1_SR', 'LANDSAT/LE07/C01/T1_SR']
    wrs2_filter = [
        {'type': 'equals', 'leftField': 'WRS_PATH', 'rightValue': 44},
        {'type': 'equals', 'leftField': 'WRS_ROW', 'rightValue': 33}]
    coll_obj = default_coll_obj(
        collections=collections,
        geometry=ee.Geometry.Rectangle(-125, 35, -120, 40),
        filter_args={c: wrs2_filter for c in collections})
    output = utils.getinfo(coll_obj._build(variables=['et']))
    assert {x[5:11] for x in parse_scene_id(output)} == {'044033'}


def test_Collection_build_invalid_variable_exception():
    """Test if Exception is raised for an invalid variable"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj()._build(variables=['FOO']))


def test_Collection_build_no_variables_exception():
    """Test if Exception is raised if variables is not set in init or method"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(variables=[])._build())


def test_Collection_overpass_default():
    """Test overpass method with default values (variables from Class init)"""
    output = utils.getinfo(default_coll_obj().overpass())
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES
    assert parse_scene_id(output) == SCENE_ID_LIST


def test_Collection_overpass_class_variables():
    """Test that custom class variables are passed through to build function"""
    output = utils.getinfo(default_coll_obj(variables=['et']).overpass())
    assert {y['id'] for x in output['features'] for y in x['bands']} == {'et'}


def test_Collection_overpass_method_variables():
    """Test that custom method variables are passed through to build function"""
    output = utils.getinfo(default_coll_obj().overpass(variables=['et']))
    assert {y['id'] for x in output['features'] for y in x['bands']} == {'et'}


def test_Collection_overpass_no_variables_exception():
    """Test if Exception is raised if variables is not set in init or method"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(variables=[]).overpass())


def test_Collection_interpolate_default():
    """Default t_interval should be custom"""
    output = utils.getinfo(default_coll_obj().interpolate())
    assert output['type'] == 'ImageCollection'
    assert parse_scene_id(output) == ['20170701']
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


def test_Collection_interpolate_variables_custom_et():
    output = utils.getinfo(default_coll_obj().interpolate(variables=['et']))
    assert {y['id'] for x in output['features'] for y in x['bands']} == {'et'}


def test_Collection_interpolate_variables_custom_ndvi():
    output = utils.getinfo(default_coll_obj().interpolate(variables=['ndvi']))
    assert {y['id'] for x in output['features'] for y in x['bands']} == {'ndvi'}


def test_Collection_interpolate_t_interval_daily():
    """Test if the daily time interval parameter works

    Since end_date is exclusive last image date will be one day earlier
    """
    coll_obj = default_coll_obj(start_date='2017-07-01', end_date='2017-07-05')
    output = utils.getinfo(coll_obj.interpolate(t_interval='daily'))
    assert output['type'] == 'ImageCollection'
    assert parse_scene_id(output)[0] == '20170701'
    assert parse_scene_id(output)[-1] == '20170704'
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


def test_Collection_interpolate_t_interval_monthly():
    """Test if the monthly time interval parameter works"""
    output = utils.getinfo(default_coll_obj().interpolate(t_interval='monthly'))
    assert output['type'] == 'ImageCollection'
    assert parse_scene_id(output) == ['201707']
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


def test_Collection_interpolate_t_interval_annual():
    """Test if the annual time interval parameter works"""
    coll_obj = default_coll_obj(start_date='2017-01-01', end_date='2018-01-01')
    output = utils.getinfo(coll_obj.interpolate(t_interval='annual'))
    assert output['type'] == 'ImageCollection'
    assert parse_scene_id(output) == ['2017']
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


def test_Collection_interpolate_t_interval_custom():
    """Test if the custom time interval parameter works"""
    output = utils.getinfo(default_coll_obj().interpolate(t_interval='custom'))
    assert output['type'] == 'ImageCollection'
    assert parse_scene_id(output) == ['20170701']
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES


# TODO: Write test for annual interpolation with a date range that is too short


# def test_Collection_interpolate_interp_days():
#     """Test if the interpolate interp_days parameter works"""
#     # Is there any way to test this without pulling values at a point?


# NOTE: For the following tests the collection class is not being
#   re-instantiated for each test so it is necessary to clear the model_args
def test_Collection_interpolate_et_reference_source_not_set():
    """Test if Exception is raised if et_reference_source is not set"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(
            et_reference_source=None, model_args={}).interpolate())


def test_Collection_interpolate_et_reference_band_not_set():
    """Test if Exception is raised if et_reference_band is not set"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(
            et_reference_band=None, model_args={}).interpolate())


def test_Collection_interpolate_et_reference_factor_not_set():
    """Test if Exception is raised if et_reference_factor is not set"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(
            et_reference_factor=None, model_args={}).interpolate())


def test_Collection_interpolate_et_reference_factor_exception():
    """Test if Exception is raised if et_reference_factor is not a number or negative"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(
            et_reference_factor=-1, model_args={}).interpolate())


# CGM - Resample is not working so commenting out for now
# def test_Collection_interpolate_et_reference_resample_not_set():
#     """Test if Exception is raised if et_reference_resample is not set"""
#     with pytest.raises(ValueError):
#         utils.getinfo(default_coll_obj(
#             et_reference_resample=None, model_args={}).interpolate())


def test_Collection_interpolate_et_reference_resample_exception():
    """Test if Exception is raised if et_reference_resample is not set"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(
            et_reference_resample='deadbeef', model_args={}).interpolate())


def test_Collection_interpolate_et_reference_params_kwargs():
    """Test setting et_reference parameters in the Collection init args"""
    output = utils.getinfo(default_coll_obj(
        et_reference_source='IDAHO_EPSCOR/GRIDMET', et_reference_band='eto',
        et_reference_factor=0.5, et_reference_resample='bicubic',
        model_args={}).interpolate())
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES
    assert output['features'][0]['properties']['et_reference_factor'] == 0.5
    assert output['features'][0]['properties']['et_reference_resample'] == 'bicubic'


def test_Collection_interpolate_et_reference_params_model_args():
    """Test setting et_reference parameters in the model_args"""
    output = utils.getinfo(default_coll_obj(
        et_reference_source=None, et_reference_band=None,
        et_reference_factor=None, et_reference_resample=None,
        model_args={'et_reference_source': 'IDAHO_EPSCOR/GRIDMET',
                    'et_reference_band': 'eto',
                    'et_reference_factor': 0.5,
                    'et_reference_resample': 'bicubic'}).interpolate())
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES
    assert output['features'][0]['properties']['et_reference_factor'] == 0.5
    assert output['features'][0]['properties']['et_reference_resample'] == 'bicubic'


def test_Collection_interpolate_et_reference_params_interpolate_args():
    """Test setting et_reference parameters in the interpolate call"""
    et_reference_args = {'et_reference_source': 'IDAHO_EPSCOR/GRIDMET',
                         'et_reference_band': 'eto',
                         'et_reference_factor': 0.5,
                         'et_reference_resample': 'bicubic'}
    output = utils.getinfo(default_coll_obj(
        et_reference_source=None, et_reference_band=None,
        et_reference_factor=None, et_reference_resample=None,
        model_args={}).interpolate(**et_reference_args))
    assert {y['id'] for x in output['features'] for y in x['bands']} == VARIABLES
    assert output['features'][0]['properties']['et_reference_factor'] == 0.5
    assert output['features'][0]['properties']['et_reference_resample'] == 'bicubic'


def test_Collection_interpolate_t_interval_exception():
    """Test if Exception is raised for an invalid t_interval parameter"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj().interpolate(t_interval='DEADBEEF'))


def test_Collection_interpolate_interp_method_exception():
    """Test if Exception is raised for an invalid interp_method parameter"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj().interpolate(interp_method='DEADBEEF'))


def test_Collection_interpolate_interp_days_exception():
    """Test if Exception is raised for an invalid interp_days parameter"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj().interpolate(interp_days=0))


def test_Collection_interpolate_no_variables_exception():
    """Test if Exception is raised if variables is not set in init or method"""
    with pytest.raises(ValueError):
        utils.getinfo(default_coll_obj(variables=[]).interpolate())


# def test_Collection_overpass_values():
#     output = utils.point_coll_value(
#         default_coll_obj().overpass(), xy=TEST_POINT, scale=30)
#     pprint.pprint(output)
#     assert False


# def test_Collection_interpolate_values_daily():
#     output = utils.point_coll_value(
#         default_coll_obj().interpolate(t_interval='daily'),
#         xy=TEST_POINT, scale=30)
#     pprint.pprint(output)
#     assert False


# def test_Collection_interpolate_values_monthly():
#     output = utils.point_coll_value(
#         default_coll_obj().interpolate(t_interval='monthly'),
#         xy=TEST_POINT, scale=30)
#     pprint.pprint(output)
#     assert False


# def test_Collection_interpolate_values_annual():
#     output = utils.point_coll_value(
#         default_coll_obj().interpolate(t_interval='annual'),
#         xy=TEST_POINT, scale=30)
#     pprint.pprint(output)
#     assert False


# def test_Collection_interpolate_values_custom():
#     output = utils.point_coll_value(
#         default_coll_obj().interpolate(t_interval='custom'),
#         xy=TEST_POINT, scale=30)
#     pprint.pprint(output)
#     assert False


# def test_Collection_interpolate_values_interp_days():
#     args = default_coll_args()
#     args['start_date'] = '2017-06-30'
#     output = utils.point_coll_value(
#         sims.Collection(**args).overpass(
#             variables=['kc', 'et_reference']), xy=TEST_POINT, scale=30)
#     pprint.pprint(output)
#
#     output = utils.point_coll_value(
#         sims.Collection(**args).interpolate(
#             variables=['kc', 'et_reference'], t_interval='daily',
#             interp_days=32),
#         xy=TEST_POINT, scale=30)
#     pprint.pprint(output)
#     assert False


def test_Collection_interpolate_output_type_default():
    """Test if output_type parameter is defaulting to float"""
    test_vars = ['et', 'et_reference', 'et_fraction', 'ndvi', 'count']
    output = utils.getinfo(default_coll_obj(variables=test_vars).interpolate())
    output = output['features'][0]['bands']
    bands = {info['id']: i for i, info in enumerate(output)}
    assert(output[bands['et']]['data_type']['precision'] == 'float')
    assert(output[bands['et_reference']]['data_type']['precision'] == 'float')
    assert(output[bands['et_fraction']]['data_type']['precision'] == 'float')
    assert(output[bands['ndvi']]['data_type']['precision'] == 'float')
    assert(output[bands['count']]['data_type']['precision'] == 'int')


# TODO: Change to 'projects/openet/crop_type/annual'
def test_Collection_interpolate_custom_model_args():
    """Test passing in a model specific parameter through model_args"""
    model_args = {'crop_type_source': 'projects/openet/crop_type_mvp'}
    output = utils.getinfo(default_coll_obj(model_args=model_args).interpolate())
    output = output['features'][0]['properties']
    assert output['crop_type_source'] == 'projects/openet/crop_type_mvp'


def test_Collection_interpolate_only_interpolate_images():
    """Test if count band is returned if no images in the date range"""
    variables = {'et', 'count'}
    output = utils.getinfo(default_coll_obj(
        collections=['LANDSAT/LC08/C01/T1_SR'],
        geometry=ee.Geometry.Point(-123.623, 44.745),
        start_date='2017-04-01', end_date='2017-04-30',
        variables=list(variables), cloud_cover_max=70).interpolate())
    pprint.pprint(output)
    assert {y['id'] for x in output['features'] for y in x['bands']} == variables