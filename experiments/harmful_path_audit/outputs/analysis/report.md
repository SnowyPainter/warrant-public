# Harmful OpenPath → Selective Suppression → Recovery Audit

Matched seeds: 5; Full wins: 4/5.

| Test | Result |
|---|---:|
| OpenPath hard attention > gold | True (0.0136 vs 0.0129) |
| OpenPath random attention > gold | True (0.0194 vs 0.0129) |
| Full hard gate < gold gate | True (0.5054 vs 0.8882) |
| Full random gate < gold gate | True (0.4665 vs 0.8882) |
| Full − OpenPath MRR | +0.0030 |
| Unsupported fraction change | -0.0056 |

## Decision

All five pre-registered conditions must be reported; a selective-suppression claim is not made if the mean recovery condition fails.
