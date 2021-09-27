import numpy as np
import torch

import dysts
from dysts.analysis import sample_initial_conditions

def get_train_test(equation_name, standardize=True, **kwargs):
    """
    Generate train and test trajectories for a given dynamical system
    """

    eq = getattr(dysts.flows, equation_name)()
    
    train_ic, test_ic = sample_initial_conditions(eq, 2)

    eq.ic = train_ic
    tpts_train, sol_train = eq.make_trajectory(
        1000, resample=True, return_times=True, **kwargs
    )
    eq.ic = test_ic
    tpts_test, sol_test = eq.make_trajectory(
        200, resample=True, return_times=True, **kwargs
    )
    
    if standardize:
        center = np.mean(sol_train, axis=0)
        scale = np.std(sol_train, axis=0)
        sol_train = (sol_train - center) / scale
        sol_test = (sol_test - center) / scale
    
    return (tpts_train, sol_train), (tpts_test, sol_test)


class BatchLoader:
    def __init__(
        self, dataset, tlen, tpts=None, replace=False, batch_size=64,
        standardize=True,
        random_state=None
    ):

        if len(dataset.shape) == 1:
            self.data = dataset[None, :, None]
        elif len(dataset.shape) == 2:
            self.data = dataset[None, ...]
        else:
            self.data = dataset
            

        (self.npts, self.maxt, self.dim) = self.data.shape

        if tpts is not None:
            self.tpts = tpts
        else:
            self.tpts = np.arange(self.maxt)

        self.tlen = tlen
        self.batch_size = batch_size
        self.random_state = random_state
        self.replace = replace
        np.random.seed(self.random_state)

        ## Prevent batches that are too large
        if self.batch_size > self.maxt - self.tlen:
            self.batch_size = self.maxt - self.tlen

        self.device = torch.device(
            "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
        )

    def get_batch(self):
        """
        Randomly sample a batch
        """
        # pick M random timepoints for the batch
        # pick M random ic points for the batch
        ic_inds = torch.from_numpy(
            np.random.choice(
                np.arange(self.npts, dtype=np.int64), self.batch_size, replace=True,
            )
        )

        time_inds = torch.from_numpy(
            np.random.choice(
                np.arange(self.maxt - self.tlen, dtype=np.int64),
                self.batch_size,
                replace=self.replace,
            )
        )

        batch_y0 = torch.from_numpy(
            np.vstack(
                [
                    self.data[ic_val, time_val].astype(np.float32)
                    for ic_val, time_val in zip(ic_inds, time_inds)
                ]
            )
        )

        batch_t = torch.from_numpy(self.tpts[:self.tlen].astype(np.float32))  # (T) ##?
        sol_batch = self.data[np.array(ic_inds)]

        batch_full = list()
        for ic_val, time_val in zip(ic_inds, time_inds):
            sub_batch = list()
            for i in range(self.tlen):
                sub_batch.append(
                    torch.from_numpy(self.data[ic_val, time_val + i].astype(np.float64))
                )
            batch_full.append(torch.stack(sub_batch, dim=0))
        batch_y = torch.transpose(torch.stack(batch_full, dim=0), 0, 1)

        return (
            batch_y0.to(self.device),
            batch_t.to(self.device),
            batch_y.to(self.device),
        )