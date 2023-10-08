from fire import Fire
import subprocess as sp 
import os
from tqdm import tqdm

MODELS = {
    'electra' : 'crystina-z/monoELECTRA_LCE_nneg31',
    't5' : 'castorini/monot5-base-msmarco',
}

def main(script : str, run_dir : str, output_dir : str, batch_size : int = 128):
    main_args = ['python', '-m', script]
    files = [f for f in os.listdir(run_dir) if f.endswith('.tsv.gz') and 'prompt' in f]

    total = len(files) * len(MODELS)

    progress_bar = tqdm(total=total)

    for file in files:
        for name, ckpt in MODELS.items(): 
            args = main_args.copy()
            output_file = os.path.join(output_dir, f'{file.replace(".tsv.gz", "")}_{name}.tsv')
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