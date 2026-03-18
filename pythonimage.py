import exiftool
import json

def get_gopro_metadata(file_path):
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(file_path)
    return metadata

# Utilisation
meta = get_gopro_metadata("GH010001.MP4")
print(json.dumps(meta[0], indent=4))