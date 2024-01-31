import music21
import os
import glob

dirname = os.path.dirname(__file__)

MUSICXML_PATH = os.path.join(dirname, 'DATA\\rachmaninov')

musicxml_files = glob.glob(os.path.join(MUSICXML_PATH + '\\musicxml', "*.mxl"))

for file in musicxml_files:
    fn_mid = os.path.join(MUSICXML_PATH, os.path.splitext(os.path.basename(file))[0] + ".mid")
    print(os.path.basename(file))


    c = music21.converter.parse(file)

    try:
        c.write('midi', fp=fn_mid)
    except music21.Music21Exception:
        continue 
    