from fire import Fire
import subprocess as sp 
import os
from tqdm import tqdm

def main(script : str, run_dir : str, output_dir : str):
    main_args = ['python', script]
    files = [f for f in os.listdir(run_dir) if f.endswith('.tsv') and not f.startswith('normal')]

    total = len(files) 

    progress_bar = tqdm(total=total)

    for file in files:
        args = main_args.copy()
        args.extend(['--run_file', os.path.join(run_dir, file)])
        args.extend(['--res_dump', output_dir])
        sp.run(args)
        progress_bar.update(1)

    return "Done!"

if __name__ == '__main__':
    Fire(main)