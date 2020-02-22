from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import abc

import numpy as np
import pandas as pd
from pathlib import Path

class WriteResultHookBase(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, split, output_path, outputs, indices, labels=None, data=None,
                 is_train=False, reference_csv_filename=None):
        pass


class DefaultWriteResultHook(WriteResultHookBase):
    def __call__(self, split, output_path, outputs, indices ,labels=None, data=None,
                 is_train=False, reference_csv_filename=None):
        output_path = Path(output_path)
        if not output_path.is_dir():
            output_path.mkdir(parents=True)
        csv_path = output_path/'result.csv'

        if csv_path.exists(): #　overwrite existing csv
            df = pd.read_csv(csv_path)
        else:
            if reference_csv_filename is not None:
                df = pd.read_csv(reference_csv_filename)
            else:
                df = pd.DataFrame(index=indices)
            df['AnomalyScore'] = np.nan
        
        df.loc[indices,'AnomalyScore'] = outputs['anomaly_score']
        df.to_csv(csv_path, index=False)

        return

