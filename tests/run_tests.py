import sys
import os
import time
import pprint
import traceback
import argparse
from subprocess import Popen


def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))


def main():
    tests = ['state', 'trace', 'model_remote', 'inference_remote', 'basic', 'all']
    try:
        parser = argparse.ArgumentParser(description='pyprob tests', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--device', help='Set the compute device (cpu, cuda:0, cuda:1, etc.)', default='cpu', type=str)
        parser.add_argument('--seed', help='Random number seed', default=None, type=int)
        parser.add_argument('--pyprob_cpp_dir', help='', default='../../pyprob_cpp/build/pyprob_cpp', type=str)
        parser.add_argument('--tests', help='Tests to run', nargs='+', required=True, choices=tests)

        opt = parser.parse_args()

        print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))

        print('Config:')
        pprint.pprint(vars(opt), depth=2, width=50)

        if 'basic' in opt.tests:
            selected_tests = ['state', 'trace', 'distributions', 'model', 'model_remote', 'diagnostics', 'dataset', 'nn', 'train', 'util']
        elif 'all' in opt.tests:
            selected_tests = ['state', 'trace', 'distributions', 'model', 'model_remote', 'diagnostics', 'dataset', 'nn', 'train', 'util', 'inference', 'inference_remote']
        else:
            selected_tests = opt.tests

        print('pyprob tests\n')
        print('Selected tests:\n{}\n'.format(selected_tests))

        processes = []
        fail = False
        for test in selected_tests:
            print('Running test: {}'.format(test))

            if test == 'model_remote':
                cmd = [os.path.join(opt.pyprob_cpp_dir, 'test_set_defaults_and_addresses'), 'ipc://@RemoteModelSetDefaultsAndAddresses']
                print('Running {}'.format(*cmd))
                p = Popen(cmd)
                processes.append(p)
            elif test == 'inference_remote':
                cmd = [os.path.join(opt.pyprob_cpp_dir, 'test_gum'), 'ipc://@GaussianWithUnknownMeanCPP']
                print('Running {}'.format(cmd))
                p = Popen(cmd)
                processes.append(p)
                cmd = [os.path.join(opt.pyprob_cpp_dir, 'test_gum_marsaglia_replacement'), 'ipc://@GaussianWithUnknownMeanMarsagliaWithReplacementCPP']
                print('Running {}'.format(cmd))
                p = Popen(cmd)
                processes.append(p)
                cmd = [os.path.join(opt.pyprob_cpp_dir, 'test_hmm'), 'ipc://@HiddenMarkovModelCPP']
                print('Running {}'.format(cmd))
                p = Popen(cmd)
                processes.append(p)
                cmd = [os.path.join(opt.pyprob_cpp_dir, 'test_branching'), 'ipc://@BranchingCPP']
                print('Running {}'.format(cmd))
                p = Popen(cmd)
                processes.append(p)

            if os.system('python test_{}.py'.format(test)) != 0:
                print('Test failed: {}'.format(test))
                fail = True
                break

        if len(processes) > 0:
            for p in processes:
                print('Killing process {}'.format(p.pid))
                p.terminate()
        print('\nTotal duration: {}'.format(days_hours_mins_secs_str(time.time() - time_start)))
        if fail:
            sys.exit(1)
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        print('Stopped.')
    except Exception:
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    time_start = time.time()
    main()
