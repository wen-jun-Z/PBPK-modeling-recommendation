#!/usr/bin/env python3

# Unicode escapes preserve historical source filenames and field names.
"""Extract the user-selected raster diagram without redrawing any foreground.

Opaque source RGB, dimensions, lettering and positions are invariant. Object
interiors are protected explicitly; low-chroma checker pixels outside those
interiors are excluded. The translucent lavender area requires local unmatting
because it was composited with the checkerboard. No resizing or regeneration.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path('/home/xuyiqian/.codex/attachments/4c63bfd7-577c-475a-ba0f-5f4bb7d14a41/ChatGPT \u56fe\u50cf 2026\u5e7410\u670821\u65e5 19_38_07.png')
OUTPUT = ROOT / 'results/figures/workflow/figure_2_blueprint_original_transparent.png'


def extract(source: Path, preview: bool = False):
    original = Image.open(source).convert('RGB')
    if original.size != (1605, 980):
        raise ValueError('Protection masks are defined for the selected 1605 × 980 original only.')
    rgb = np.asarray(original)
    f = rgb.astype(np.float32)
    lum = f.mean(axis=2)
    chroma = np.ptp(f, axis=2)
    # All photographic/gray illustration interiors and intentionally filled
    # cards are protected, rather than deleting a global range of gray colors.
    protected = Image.new('L', original.size, 0)
    d = ImageDraw.Draw(protected)
    cards = [
        ((43,20,430,63),21), ((948,22,1438,63),20),
        ((43,538,378,580),20), ((922,542,1246,583),20),
        ((501,96,565,151),12), ((611,97,673,151),10),
        ((501,182,566,236),11), ((611,183,673,236),11),
        ((802,114,913,190),15), ((1344,94,1557,278),15),
        ((46,392,402,433),12), ((454,403,749,439),13),
        ((617,633,706,734),10), ((742,633,858,738),11),
        ((635,782,815,832),10), ((1489,637,1558,728),5),
        ((1311,847,1498,886),5), ((1513,836,1552,900),5),
        ((1265,485,1344,523),11),
    ]
    for box, radius in cards:
        d.rounded_rectangle(box, radius=radius, fill=255)
    # The paper stack, rather than its surrounding bounding rectangle.
    d.polygon([(239,143),(248,141),(247,133),(258,130),(262,138),
               (305,129),(310,137),(366,150),(349,266),(286,255),
               (264,249),(238,236)], fill=255)
    d.polygon([(229,142),(303,128),(324,240),(250,256)],fill=255)
    d.polygon([(240,129),(251,126),(255,141),(243,144)],fill=255)
    # Human silhouette, including pale anatomical details; arm gaps stay open.
    d.polygon([(111,88),(103,92),(100,101),(101,116),(107,128),
               (103,135),(80,142),(70,152),(64,176),(58,200),
               (54,220),(47,241),(43,258),(49,254),(49,269),
               (53,271),(57,257),(57,272),(62,270),(66,252),
               (64,237),(72,216),(79,186),(84,182),(85,211),
               (80,243),(81,273),(85,290),(107,290),(111,268),
               (114,268),(118,290),(138,290),(143,270),(143,242),
               (137,218),(139,196),(144,181),(150,201),(155,227),
               (162,241),(163,254),(168,267),(172,269),(169,253),
               (177,266),(179,265),(173,249),(179,252),(171,231),
               (166,213),(163,196),(159,180),(156,159),(144,146),
               (124,138),(121,127),(128,115),(129,102),(124,93),
               (117,89)], fill=255)
    # Native bottle interiors and other small neutral foreground areas.
    d.rounded_rectangle((226,222,256,258),radius=7,fill=255)
    d.ellipse((226,189,257,201),fill=255)
    d.rounded_rectangle((46,697,79,746),radius=7,fill=255)
    d.ellipse((49,675,76,685),fill=255)
    d.polygon([(714,194),(724,198),(734,194),(739,201),(733,210),
               (729,213),(731,238),(722,238),(724,212),(717,208)],fill=255)
    # Conditional decision diamond.
    d.polygon([(540,651),(607,720),(541,784),(466,723)],fill=255)
    # Locked model tablet, preserving all pale diagram lines and gray details.
    d.polygon([(919,638),(1028,625),(1042,723),(934,738)],fill=255)
    d.rounded_rectangle((1020,659,1046,687),radius=4,fill=255)
    # IV bag / target human, report and stop symbols.
    d.rounded_rectangle((1181,621,1193,647),radius=2,fill=255)
    d.ellipse((1127,612,1152,639),fill=255)
    d.polygon([(1132,635),(1147,635),(1149,643),(1159,648),
               (1163,690),(1157,691),(1153,659),(1150,691),
               (1131,691),(1128,661),(1124,692),(1120,690),
               (1125,650),(1134,644)],fill=255)
    d.ellipse((1531,861,1561,891),fill=255)
    d.ellipse((646,790,677,821),fill=255)
    # Preserve solid AUC fills, but not the checkerboard above the curves.
    lavender=Image.new('L',original.size,0)
    ImageDraw.Draw(lavender).polygon(
        [(244,784),(249,764),(255,748),(262,733),(270,723),
         (281,720),(290,726),(307,740),(325,752),(346,763),
         (370,773),(399,780),(432,784)],fill=255)
    d.polygon([(1306,746),(1311,719),(1318,690),(1326,668),
               (1338,658),(1352,671),(1369,690),(1387,706),
               (1406,718),(1428,730),(1450,736),(1450,747)],fill=255)
    for x,y,r in [(1014,118,6),(1088,204,6),(1141,222,6),
                  (1205,251,6),(1283,247,6)]:
        d.ellipse((x-r,y-r,x+r,y+r),fill=255)
    p = np.asarray(protected) > 0

    # Keep colored and dark cores, then only adjacent faint antialiasing.
    # Background tint/noise alone must not become an opaque foreground island.
    core = (chroma >= 27) | (lum <= 88)
    near = ndi.binary_dilation(core, iterations=2)
    edge_color = np.clip((chroma-9)/15,0,1)
    edge_dark = np.clip((129-lum)/37,0,1)
    edge_white = np.clip((lum-222)/20,0,1)
    alpha = np.maximum.reduce([edge_color,edge_dark,edge_white]) * near
    alpha[core | p] = 1
    # Recover only the already-present translucent purple tint; the opaque
    # curve/axes/points retain every original RGB pixel. A gray background
    # contributes equally to R/G/B, so B-G isolates the purple contribution.
    semi=(np.asarray(lavender)>0) & (chroma<65) & (lum>130)
    tint_alpha=ndi.gaussian_filter(np.maximum(f[:,:,2]-f[:,:,1],0)/125,0.65)
    alpha[semi]=np.clip(tint_alpha[semi],0,0.4)
    alpha8 = np.round(alpha*255).astype(np.uint8)
    rgba = np.dstack([rgb,alpha8])
    rgba[semi,:3]=[116,55,180]
    assert np.array_equal(rgba[:,:,:3][~semi],rgb[~semi])
    out = Image.fromarray(rgba)
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    out.save(OUTPUT)
    assert out.getchannel('A').getextrema() == (0,255)
    print({'output':str(OUTPUT),'size':out.size,'opaque_RGB_max_difference':0,
           'transparent_fraction':round(float((alpha8==0).mean()),4),
           'protected_pixels':int(p.sum()),'translucent_unmatting_pixels':int(semi.sum())})
    if preview:
        folder=ROOT/'runtime/presentation_build/original_blueprint_extraction'
        folder.mkdir(parents=True,exist_ok=True)
        for name,color in [('white','#FFFFFF'),('dark','#232D3B')]:
            bg=Image.new('RGBA',out.size,color)
            bg.alpha_composite(out)
            bg.convert('RGB').save(folder/f'{name}.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    extract(args.source,args.preview)
