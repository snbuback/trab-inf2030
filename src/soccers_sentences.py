'''
Created on Aug 29, 2010

@author: snbuback
'''
import os
from lxml import etree

DIRETORIO = "%s/Dropbox/puc-rj/trab/banco/" % (os.getenv('HOME'))
classes_aceitas = ['penalti', 'gol', 'cartao', 'substituicao', 'narracao']

def read_files():
    if '_cache' in dir(read_files):
        return read_files._cache

    featuresset = {}
    for arq in os.listdir(DIRETORIO):
        if arq.endswith(".xml"):
            root = etree.parse(DIRETORIO + arq)
            for element in root.iter():
                tag = element.tag
                sentence = element.text
                if tag in classes_aceitas:
                    if tag in featuresset:
                        featuresset[tag] += [sentence]
                    else:
                        featuresset[tag] = [sentence]

    print "estatisticas"
    total = 0
    for tag in featuresset:
	print tag, " = ", len(featuresset[tag])
	total += len(featuresset[tag])
    print "total = %d" % total

    read_files._cache = featuresset
    return featuresset

if __name__ == '__main__':
    f = read_files()
    for t in f:
        print t, len(f[t])
