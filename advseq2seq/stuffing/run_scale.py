from fire import Fire
import subprocess as sp 
import os
from tqdm import tqdm

MODELS = {
    't5.small' : 'castorini/monot5-small-msmarco-100k',
    't5.large' : 'castorini/monot5-large-msmarco',
    't5.3b' : 'castorini/monot5-3b-msmarco',
}

def main(script : str, run_dir : str, output_dir : str, name : str = 't5.3b', batch_size : int = None):
    BATCH_SIZES = {
    't5.small' : 256,
    't5.large' : 256,
    't5.3b' : 24,
    }
    main_args = ['python', script]
    files = [f for f in os.listdir(run_dir)]

    total = len(files) 

    progress_bar = tqdm(total=total)
    if batch_size is None: batch_size = BATCH_SIZES[name]

    for file in files:
        ckpt = MODELS[name]
        args = main_args.copy()
        output_file = os.path.join(output_dir, f'{file.replace(".tsv", "")}_{name}.tsv')
        if os.path.exists(output_file):
            print('Skipping ', output_file, ' as it already exists')
            progress_bar.update(1)
            continue
        args.extend(['--run_file', os.path.join(run_dir, file)])
        args.extend(['--output_file', output_file])
        args.extend(['--model', name])
        args.extend(['--model_name_or_path', ckpt])
        args.extend(['--batch_size', str(batch_size)])
        sp.run(args)
        progress_bar.update(1)

    return "Done!"

if __name__ == '__main__':
    Fire(main)