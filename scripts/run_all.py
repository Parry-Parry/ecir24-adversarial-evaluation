from fire import Fire
import subprocess as sp 
import os

MODELS = {
    't5' : 'castorini/monot5-base-msmarco',
    'electra' : 'crystina-z/monoELECTRA_LCE_nneg31'
}

def main(script : str, run_dir : str, output_dir : str):
    main_args = ['python', script]
    files = [f for f in os.listdir(run_dir) if f.endswith('.tsv')]

    for file in files:
        for name, ckpt in MODELS.items(): 
            args = main_args.copy()
            output_file = os.path.join(output_dir, f'{file.replace(".tsv", "")}_{name}.tsv')
            args.extend(['--run_file', os.path.join(run_dir, file)])
            args.extend(['--output_file', output_file])
            args.extend(['--model', name])
            args.extend(['--model_name_or_path', ckpt])
            sp.run(args)

    return "Done!"

if __name__ == '__main__':
    Fire(main)