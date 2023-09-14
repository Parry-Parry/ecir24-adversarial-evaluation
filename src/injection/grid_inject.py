from fire import Fire
import subprocess as sp

def main(script : str, token_file : str, doc_file : str, output_dir : str, n : int = None, mode : str = None):
    N = [1, 2, 3, 4, 5]
    modes = ['random', 'start', 'end']
    if n: N = [n]
    if mode: modes = [mode]
    main_args = ['python', script, '--token_file', token_file, '--doc_file', doc_file, '--output_dir', output_dir]

    for n in N:
        for mode in modes:
            args = main_args + ['--n', str(n), '--mode', mode]
            sp.run(args)
    return "Done!"

if __name__ == '__main__':
    Fire(main)