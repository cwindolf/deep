from streetview.gsv_api_env import KEY
import googlemaps
from numpy.random import uniform, randint
import requests

PVD_SE_LAT, PVD_SE_LONG = 41.769634, -71.373618
PVD_NW_LAT, PVD_NW_LONG = 41.865329, -71.479877
SV_API_URL = 'https://maps.googleapis.com/maps/api/streetview?size=1280x800&location=%s,%s&fov=75&heading=%d&key=' + KEY
SV_METADATA_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata?location=%s,%s&key=' + KEY
replacements = (('Ave', 'Avenue'), ('Av', 'Avenue'), ('Bvd', 'Boulevard'),
                ('Rd', 'Road'), ('St', 'Street'), ('Ln', 'Lane'), ('Ct', 'Court'),
                ('Dr', 'Drive'), ('W ', 'West '), ('E ', 'East '), ('Blvd', 'Boulevard'),
                ('N ', 'North '), ('S ', 'South '), ('Mt', 'Mount'),
                ('&', 'and'), ('Expy', 'Expressway'), ('Pkwy', 'Parkway'))


def random_providence_image(output_filename):
    '''
    Write a random image in PVD to `output_filename`, and return
    a title for that image (the address.)
    '''
    lat, lng = random_point_in_providence()
    url = SV_API_URL % (lat, lng, randint(0, 360))
    r = requests.get(url, stream=True)
    with open(output_filename, 'wb') as out_jpg:
        for chunk in r.iter_content(chunk_size=256):
            out_jpg.write(chunk)
    return address(lat, lng)


def random_point_in_providence():
    lat, lng = random_point_in_pvd_square()
    while not point_is_in_providence(lat, lng) or point_has_no_image(lat, lng):
        lat, lng = random_point_in_pvd_square()
    return lat, lng


def random_point_in_pvd_square():
    return uniform(PVD_SE_LAT, PVD_NW_LAT), uniform(PVD_NW_LONG, PVD_SE_LONG) 


def point_is_in_providence(lat, lng):
    gmaps = googlemaps.Client(key=KEY)
    addr = gmaps.reverse_geocode((lat, lng))[0]['address_components']
    for component in addr:
        if 'locality' in component['types']:
            if component['long_name'].lower().strip() == 'providence':
                return True
    return False


def point_has_no_image(lat, lng):
    return requests.get(SV_METADATA_URL % (lat, lng)).json()['status'] == 'ZERO_RESULTS'


def address(lat, lng):
    gmaps = googlemaps.Client(key=KEY)
    addr = gmaps.reverse_geocode((lat, lng))[0]['formatted_address'].split(',')[0]
    for rep, orig in replacements:
        if orig not in addr:
            addr = addr.replace(rep, orig)
    return addr

'''
Refs:
Random points:
    using lat/long in a circle:
    http://gis.stackexchange.com/questions/25877/how-to-generate-random-locations-nearby-my-location
    in arbitrary polygons:
    http://gis.stackexchange.com/questions/6412/generate-points-that-lie-inside-polygon
'''
print(random_providence_image('out.jpg'))
