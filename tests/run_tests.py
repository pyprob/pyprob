import sys
import os
import time
import pprint
import traceback
import argparse
import unittest

def days_hours_mins_secs_str(total_seconds):
    d, r = divmod(total_seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return '{0}d:{1:02}:{2:02}:{3:02}'.format(int(d), int(h), int(m), int(s))

def main():
    try:
        parser = argparse.ArgumentParser(description='pyprob tests', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--device', help='Set the compute device (cpu, cuda:0, cuda:1, etc.)', default='cpu', type=str)
        parser.add_argument('--seed', help='Random number seed', default=None, type=int)
        parser.add_argument('--tests', help='Tests to run', nargs='+', required=True, choices=['state', 'trace'])

        opt = parser.parse_args()

        print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))

        print('Config:')
        pprint.pprint(vars(opt), depth=2, width=50)

        print('pyprob tests\n')
        print('Selected tests:\n{}\n'.format(opt.tests))

        for test in opt.tests:
            print('Running test: {}'.format(test))
            if os.system('python test_{}.py'.format(test)) != 0:
                print('Test failed: {}'.format(test))
                sys.exit(1)

        print('\nTotal duration: {}'.format(days_hours_mins_secs_str(time.time() - time_start)))
        sys.exit(0)

    except KeyboardInterrupt:
        print('Stopped.')
    except Exception:
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    time_start = time.time()
    main()
