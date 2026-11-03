# Baselines

Each method was run from its official repository at a pinned commit. The table gives our 3-D adaptation for each, as a patch
against that commit and/or our driver scripts. Run configs: `configs/<folder>_<test>.sh` (`configs/common.sh` holds the shared
settings).

| method | folder | official repository | commit | our change |
|---|---|---|---|---|
| LUNA (3-D) | `luna3d/` | https://github.com/mlbio-epfl/LUNA | 64f10eb55a4cdc862c0535051a1fe042ab2ceadc | `luna_3d.patch` + drivers |
| REMAP | `remap/` | https://github.com/ShunzhouJiang/REMAP | b663e2ba21d93a8451e088336fd3fdc32633ae5b | none to the method; drivers |
| CeLEry (3-D) | `celery3d/` | https://github.com/QihuangZhang/CeLEry | b0b73764bb517b0b7b360d4b18bed5d6b77a4615 | `celery_3d.patch` + drivers |
| Tangram (3-D) | `tangram3d/` | https://github.com/broadinstitute/Tangram | 4c68995a418f41dc8caef567598c4d9b47781a13 | none to the method; drivers |
| novoSpaRc (3-D) | `novosparc3d/` | https://github.com/rajewsky-lab/novosparc | a3c9f3d76e59c7a601cfcd0e211630e2458d79a3 | none to the method; drivers |

Each folder's README explains the adaptation, the inputs, and how each run of the paper was configured.
