"""Decode one API image and save in the existing Stage3 PNG/prompt layout."""
import base64
import io

from PIL import Image

from stage3_image_generation.pipeline import completed


def saved_image_config(config):
    g = config['generation']
    width, height = config['save_size'] or (g['width'], g['height'])
    return {'generation': {'width': width, 'height': height}}


def is_saved(directory, row, config):
    return completed(directory, row, saved_image_config(config))


def decode_image(body, config):
    if not isinstance(body, dict) or body.get('error'):
        raise ValueError('Image response contains an error or is not an object')
    data = body.get('data')
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError('Expected exactly one image in response.data')
    encoded = data[0].get('b64_json')
    if not isinstance(encoded, str) or not encoded:
        raise ValueError('Expected a base64 image in data[0].b64_json')
    raw = base64.b64decode(encoded, validate=True)
    g = config['generation']
    expected_format = {'png': 'PNG', 'jpeg': 'JPEG', 'webp': 'WEBP'}[g['output_format']]
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != expected_format or image.size != (g['width'], g['height']):
            raise ValueError('Returned image format or dimensions do not match the submitted request')
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        image = image.convert('RGBA' if 'A' in image.getbands() or 'transparency' in image.info else 'RGB')
        target = config['save_size']
        if target is not None and image.size != tuple(target):
            image = image.resize(tuple(target), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format='PNG')
        return output.getvalue()
