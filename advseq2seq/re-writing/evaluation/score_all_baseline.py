import os
import subprocess as sp
from fire import Fire
from tqdm import tqdm

MODELS = {
    'electra' : 'crystina-z/monoELECTRA_LCE_nneg31',
    't5.base' : 'castorini/monot5-base-msmarco',
    't5.small' : 'castorini/monot5-small-msmarco-100k',
    't5.large' : 'castorini/monot5-large-msmarco',
    't5.3b' : 'castorini/monot5-3b-msmarco',
    'tasb' : 'sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco',
    'colbert' : 'http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip',
    'bm25' : 'na'
}



def main(script : str, file : str, output_dir : str, batch_size : int = 128):
    main_args = ['python', '-m', script]

    total = len(MODELS)

    progress_bar = tqdm(total=total)
    
    for name, ckpt in MODELS.items(): 
        args = main_args.copy()
        output_file = os.path.join(output_dir, f'normal_{name}.tsv')
        if os.path.exists(output_file):
            print('Skipping ', output_file, ' as it already exists')
            progress_bar.update(1)
            continue
        args.extend(['--run_file', file])
        args.extend(['--output_file', output_file])
        args.extend(['--model', name])
        args.extend(['--model_name_or_path', ckpt])
        args.extend(['--batch_size', str(batch_size)])
        sp.run(args)
        progress_bar.update(1)

    return "Done!"

if __name__ == '__main__':
    Fire(main)