import logging
import numpy as np


def check_blas_config():
    """ checks if using OpenBlas/Intel MKL
        This function directly adopted from
        https://github.com/benfred/implicit/blob/master/implicit/utils.py
    """
    pkg_dict = {'OPENBLAS':'openblas', 'MKL':'blas_mkl'}
    for pkg, name in pkg_dict.items():
        if (np.__config__.get_info('{}_info'.format(name))
            and
            os.environ.get('{}_NUM_THREADS'.format(pkg)) != '1'):
            logging.warning(
                "{} detected, but using more than 1 thread. Its recommended "
                "to set it 'export {}_NUM_THREADS=1' to internal multithreading"
                .format(name, name)
            )
