import bz2
import pickle
import _pickle as cPickle

def create_pickle(title, data):
    '''
    Create a pickle object to save large files.
    Params:
    :title: file to be saved in a pickle
    :data: output pickle
    '''
    with open(title, 'wb') as f:
        pickle.dump(data, f)

def compressed_pickle(title, data):
    '''
    Compress an object into a pickle to reduce the size.
    Params:
    :title: file to be compressed in a pickle
    :data: output compressed pickle
    '''
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)
        
def decompress_pickle(file):
    '''
    Decompress a compressed pickle.
    Params:
    :file: file to be decompressed
    '''
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data
