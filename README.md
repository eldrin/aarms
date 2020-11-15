# ***AARMS*** : Attribute-Aware Recommender ModelS

[![Build Status](https://travis-ci.com/eldrin/aarms.svg?branch=master)](https://travis-ci.com/eldrin/aarms)

It aims for providing a set of attribute-aware recommender models running in reasonably fast execution time. Right now, the main driver is a linear model using Alternating Least Square (ALS) algorithm, which allows to plug in diverse form of side information in various format (user/item similarity, dense feature, sparse feature).


## Getting Started

It's not on PyPI yet, means you need to install the package using the `pip` and `git`

```console
$ pip install git+https://github.com/eldrin/aarms.git@master
```

### Quick Look

```python
from aarms.models import ALS

n_components = 5
lmbda = 1  # loss weight per each side information
als = ALS(n_components)

# Assume data loaded
# ===================
# `user_item` is (N x M) user-item interaction matrix
# `user_user` is (N x N) user-user similarity sparse matrix
# `user_dense_feature` is (N x D) user feature dense matrix
# `item_other` is (M x R) item-other entitiy (i.e. tag) sparse matrix
# `item_sparse_feature` is (M x S) item feature sparse matrix

# fit factors
als.fit(
  user_item,
  user_user = user_user,
  user_dense_feature = user_dense_feature,
  item_other = item_other,
  item_sparse_feature = user_sparse_feature,
  lmbda_user_user = lmbda,
  lmbda_user_dense_feature = lmbda,
  lmbda_item_other = lmbda,
  lmbda_item_sparse_feature = lmbda
)
```

## Current Status & Contributing

As a pre-alpha version, currently, we mostly provide the API specialized on the recommendation. However, we plan to extend the API as general as possible soon. If interested, feel free to send pull requests and drop issues. We are more than happy to listen to your thoughts and ideas.

## Authors

- Jaehun Kim

## TODOs

- [ ] Symmetric ALS (with features)
- [ ] refactoring and code wrangling
- [ ] code coverage
  - [ ] adding more tests

## License


This project is licensed under the MIT License - see the LICENSE.md file for details


## Reference

This package refers much to [implicit](https://github.com/benfred/implicit) package. Please check the package if you are looking for feature-rich, more scalable version supporting the user-item matrix decomposition for recommender system!
