# Main-budget path/permission decomposition

All entries use the main-benchmark protocol, the full configured data budget,
30 epochs, and seeds 7/17/37. Values are mean primary metric across three seeds.

## CTDG (AUC)

| Dataset | Backbone | Base | OpenPath (`g=1`) | Full | OpenPath−Base | Full−OpenPath |
|---|---:|---:|---:|---:|---:|---:|
| LastFM | DyGFormer | 0.848165 | 0.900978 | 0.902252 | +0.052813 | +0.001275 |
| LastFM | GraphMixer | 0.880555 | 0.914216 | 0.913940 | +0.033661 | -0.000276 |
| LastFM | TGAT | 0.859843 | 0.889831 | 0.889664 | +0.029988 | -0.000167 |
| MOOC | DyGFormer | 0.976259 | 0.976770 | 0.977308 | +0.000511 | +0.000538 |
| MOOC | GraphMixer | 0.976993 | 0.977996 | 0.977386 | +0.001003 | -0.000610 |
| MOOC | TGAT | 0.963883 | 0.971560 | 0.971508 | +0.007677 | -0.000052 |
| Wikipedia | DyGFormer | 0.944008 | 0.982261 | 0.983634 | +0.038253 | +0.001373 |
| Wikipedia | GraphMixer | 0.952897 | 0.983385 | 0.983216 | +0.030488 | -0.000169 |
| Wikipedia | TGAT | 0.943672 | 0.982528 | 0.982377 | +0.038856 | -0.000151 |

Across the 27 matched seed-level comparisons, mean OpenPath−Base is +0.025917
and mean Full−OpenPath is +0.000196 (14/27 positive). The CTDG improvement is
therefore predominantly path exposure. The learned gate adds a small consistent
increment for DyGFormer, but not for GraphMixer or TGAT.

## TKG (MRR)

| Dataset | Backbone | Base | OpenPath (`g=1`) | Full | OpenPath−Base | Full−OpenPath |
|---|---:|---:|---:|---:|---:|---:|
| GDELT | CyGNet | 0.147802 | 0.145332 | 0.147829 | -0.002471 | +0.002497 |
| GDELT | RE-NET | 0.127962 | 0.136276 | 0.139482 | +0.008314 | +0.003206 |
| GDELT | xERTE | 0.127747 | 0.138919 | 0.142860 | +0.011172 | +0.003941 |
| ICEWS14 | CyGNet | 0.219513 | 0.210187 | 0.219551 | -0.009326 | +0.009364 |
| ICEWS14 | RE-NET | 0.207563 | 0.214102 | 0.212745 | +0.006539 | -0.001357 |
| ICEWS14 | xERTE | 0.209637 | 0.218335 | 0.217618 | +0.008697 | -0.000717 |
| ICEWS18 | CyGNet | 0.153604 | 0.143816 | 0.150146 | -0.009788 | +0.006330 |
| ICEWS18 | RE-NET | 0.142254 | 0.152030 | 0.154988 | +0.009775 | +0.002959 |
| ICEWS18 | xERTE | 0.151454 | 0.156962 | 0.157959 | +0.005508 | +0.000996 |

Across the 27 matched seed-level comparisons, mean OpenPath−Base is +0.003158
and mean Full−OpenPath is +0.003024 (21/27 positive). Thus the aggregate TKG
gain is split almost evenly between exposing the path and gating it. CyGNet is
the clearest suppression case: the ungated open path is harmful on every dataset,
while the learned permission gate recovers most or all of that damage.

## Consequence for the paper

Do not claim that permission is the dominant source of improvement on every
task/backbone. Claim a conditional mechanism: path exposure supplies usable
signal when the path is benign, whereas permission is important when exposing
the path also admits harmful contributions. CTDG establishes the former regime;
TKG, especially CyGNet, establishes the latter. Existing HotpotQA suppression
and evidence-alignment interventions should be presented as the item-level
mechanistic evidence for the same claim.

## STPP (location RMSE; lower is better)

| Dataset | Backbone | Base | OpenPath (`g=1`) | Full | Path improvement | Gate improvement |
|---|---:|---:|---:|---:|---:|---:|
| Earthquake | DeepSTPP | 1.138271 | 1.054752 | 1.050182 | +0.083519 | +0.004570 |
| Earthquake | NSTPP | 1.158937 | 1.077362 | 1.115946 | +0.081575 | -0.038585 |
| Earthquake | Transformer-STPP | 1.114705 | 1.075352 | 1.074828 | +0.039353 | +0.000524 |
| Gowalla | DeepSTPP | 0.048149 | 0.048379 | 0.048374 | -0.000230 | +0.000004 |
| Gowalla | NSTPP | 0.048550 | 0.047452 | 0.047379 | +0.001098 | +0.000073 |
| Gowalla | Transformer-STPP | 0.048346 | 0.046031 | 0.046031 | +0.002314 | +0.000000 |

STPP is mostly path-dominant. Permission is neutral or modestly helpful in five
rows and materially harmful for Earthquake/NSTPP.

## MTPP (mark MRR)

| Dataset | Backbone | Base | OpenPath (`g=1`) | Full | OpenPath−Base | Full−OpenPath |
|---|---:|---:|---:|---:|---:|---:|
| Retweets | AttNHP | 0.757374 | 0.763075 | 0.765264 | +0.005701 | +0.002189 |
| Retweets | SAHP | 0.754113 | 0.757656 | 0.757771 | +0.003543 | +0.000115 |
| Retweets | THP | 0.758386 | 0.746027 | 0.749741 | -0.012358 | +0.003714 |
| StackOverflow | AttNHP | 0.605086 | 0.603966 | 0.604066 | -0.001120 | +0.000100 |
| StackOverflow | SAHP | 0.607567 | 0.606313 | 0.606238 | -0.001254 | -0.000075 |
| StackOverflow | THP | 0.606621 | 0.607696 | 0.607556 | +0.001075 | -0.000140 |

Retweets supports a separate gate contribution for every backbone. The THP row
is another suppression case: an ungated path is harmful and permission recovers
part of the loss. StackOverflow is effectively a null regime at this precision.

## RAG (answer accuracy)

| Dataset | Backbone | Base | OpenPath (`g=1`) | Full | OpenPath−Base | Full−OpenPath |
|---|---:|---:|---:|---:|---:|---:|
| HotpotQA | FiD | 0.655460 | 0.656458 | 0.657360 | +0.000998 | +0.000902 |
| HotpotQA | LED | 0.666890 | 0.670100 | 0.667047 | +0.003210 | -0.003053 |

FiD splits its gain nearly evenly between path exposure and permission. LED is
path-dominant and its learned gate is harmful under this main-benchmark setup.
The separate evidence-labeled HotpotQA intervention remains necessary to support
the fine-grained suppression interpretation; answer accuracy alone cannot do so.

## Overall interpretation

The new control was run for every main-benchmark combination: 32 dataset/model
cells and 96 newly trained seed-level runs. It rules out a universal account in
either direction. Path exposure explains most CTDG/STPP gains, whereas learned
permission makes a separate contribution in TKG (mean +0.003024 after opening
the path; 21/27 seed-level comparisons positive), all Retweets backbones, and
FiD. Harmful-open-path recovery is visible for all three CyGNet datasets and
Retweets/THP. The revised paper should make this heterogeneity a result and state
the permission claim conditionally, rather than imply universal gate dominance.
