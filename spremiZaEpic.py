from visualization import postProcessing
from napravi_parove import parovi
from edge import sed_ivice


postProcessing('fildevi/no_bcd_000002_1.npy', 'fildevi/no_bcd_000002_backwards.npy', 'sredjeni_flow.npy')
parovi('sredjeni_flow.npy', "parovi.txt")
sed_ivice('../data_scene_flow/training/image_2/000002_11.png', "ivice.bin")
