# Permission Derivative Demo

This script compares Warrant value gating with attention-logit gating.

| Quantity | Value |
| --- | ---: |
| Sum off-diagonal derivative, Warrant | 0.000000 |
| Sum off-diagonal derivative, attention-logit gate | 0.448797 |

Warrant has diagonal item-wise permission derivatives. Attention-logit gating has non-zero off-diagonal derivatives because softmax re-normalizes mass across items.

## Alpha and Mass

| Item | alpha | gate | alpha*g | softmax(score+log gate) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.570944 | 0.900000 | 0.513850 | 0.708434 |
| 1 | 0.256542 | 0.600000 | 0.153925 | 0.212213 |
| 2 | 0.115272 | 0.400000 | 0.046109 | 0.063569 |
| 3 | 0.057242 | 0.200000 | 0.011448 | 0.015784 |
