from flask import Flask, request
import logging
import time
import subprocess
import os

LOGGING_LEVEL = logging.INFO
logging.basicConfig(format="%(asctime)-15s - %(levelname)s: %(message)s", level=LOGGING_LEVEL)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def amr_sim():
    '''
    Takes in a json request in the format
    {
        'kernel': 'wwlk', 
        'amr1': '(vv1 / bake :ARG0 (vv2 / man :mod (vv3 / big)))',
        'amr2': '(vv1 / bake :ARG0 (vv2 / woman))',
        'config_filepath': 'embedding_config.yaml'
    }

    Returns similarity score
    {
        'score': 0.4284
    }
	'''
    
    start = time.time()
    info = request.json
    
    kernel = info['kernel']
    amr1 = info['amr1']
    amr2 = info['amr2']

    if kernel == 'wwlk-theta' or kernel == 'random-walk':
        logging.warning(f"Kernel '{kernel}' is unimplemented. Similarity score will be based on WWLK.")
        file_to_run = '../src/main_wlk_wasser.py'
    elif kernel == 'wlk':
        file_to_run = '../src/main_wlk.py'
        logging.info(f"Running similarity kernel '{kernel}' using file '{file_to_run}'")
    else:
        file_to_run = '../src/main_wlk_wasser.py'
        logging.info(f"Running similarity kernel '{kernel}' using file '{file_to_run}'")
        try:
            config_filepath = info['config_filepath']
            logging.info(f"Using embeddings specified in file '{config_filepath}'")
        except:
            config_filepath = None
            logging.warning(f"No embedding configurations specified. Edge embeddings will be randomly instantiated, and node embeddings will come from Glove.")
    
    # prep files for ingestion into wwlk script

    amr1file = open('amr1.txt', 'w')
    amr1file.write(amr1)
    amr1file.close()
    amr1filesize = os.path.getsize('amr1.txt')

    amr2file = open('amr2.txt', 'w')
    amr2file.write(amr2)
    amr2file.close()
    amr2filesize = os.path.getsize('amr2.txt')

    if config_filepath:
        process = subprocess.run(['python', file_to_run, '-a', 'amr1.txt', '-b', 'amr2.txt', '-embedding_config_file', config_filepath], capture_output=True)    
    else:
        process = subprocess.run(['python', file_to_run, '-a', 'amr1.txt', '-b', 'amr2.txt'], capture_output=True)

    result = float(process.stdout.decode('UTF-8').strip())

    stop = time.time()
    logging.info(f"Similarity between AMR1 ({amr1filesize} bytes) and AMR2 ({amr2filesize} bytes) generated in {stop - start} seconds.")
    
    os.remove('amr1.txt')
    os.remove('amr2.txt')

    return {'score': result}

