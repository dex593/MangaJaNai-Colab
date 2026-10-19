# GFISRV2 source

`gfisrv2.py` is copied without changes from `rewaifu/resselt`, commit
`4a0c2c163f0ba98b7c548221590bec27fe80a861`, path
`resselt/archs/gfisrv2/arch.py` (resselt 1.4.1).

Source: https://github.com/rewaifu/resselt/blob/4a0c2c163f0ba98b7c548221590bec27fe80a861/resselt/archs/gfisrv2/arch.py

Copyright 2025 rewaifu organization. MIT license and original acknowledgements
are preserved in `LICENSE-resselt.md`. The GFISRV2 model author is Umzi.
The separately downloaded MangaScale weights are CC BY 4.0 according to the
model announcement supplied by the user; this is distinct from the source license.

Only this self-contained architecture is included, so installing the project
does not install resselt's full model library or change the runtime's Torch and
safetensors versions. `manga_colab/extra_arches.py` supplies a Spandrel descriptor
for the verified default 4x RGB MangaScale model. No forward operations are changed.
