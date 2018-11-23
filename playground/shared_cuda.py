import torch.multiprocessing as mp

from torch.autograd import Variable
import torch
import numpy as np
import time

import torch.multiprocessing as mp
print(mp.get_all_start_methods())
def worker(tensor):
    while True:
        tensor += 0.01
        time.sleep(1)


def main():
    mp.set_start_method('spawn')
    tt = np.ones((100, 100, 100))

    t = torch.from_numpy(tt)
    t = t.share_memory_()  # share 1
    t = t.cuda(async=True)
    # t=t.share_memory_() #share 2. did not work.

    processes = []
    for i in range(10):
        p = mp.Process(target=worker, args=(t,))
        p.daemon = True
        p.start()
        processes.append(p)

    print('running')
    time.sleep(10)
    print(t)


if __name__ == '__main__':
    main()