# coding: utf-8

from .deepspeech import CTCModel

def build_model(charmap, cfg):
    return eval(f"{cfg['class']}(charmap, **cfg['params'])")
