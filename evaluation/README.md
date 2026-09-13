# Fixed evaluation images

`manifest.json` defines four unchanged scikit-image v0.25.2 images with human
labels, source URLs, redistribution permissions, and SHA-256 checksums.
The images are committed so evaluation never downloads a changing dataset.

Chelsea is CC0 by Stefan van der Walt; coffee is CC0 by Rachel Michetti, courtesy
of Pikolo Espresso Bar; camera is CC0 by Lav Varshney; the clock was released into
the public domain by Stefan van der Walt. The permissions are documented in
[scikit-image's official data documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html).
The [CC0 dedication](https://creativecommons.org/publicdomain/zero/1.0/) permits
redistribution and modification. The camera image is the replacement introduced
in scikit-image 0.18, not the earlier image with copyright restrictions.

Labels describe the visible objects before running the model. Multiple accepted
labels are documented where ImageNet divides domestic cats into visually
overlapping categories, or the photograph has two applicable object labels.
An image counts once, regardless of the number of accepted labels. We do not
relabel a model mistake to make the gate pass.

The required top-1 and top-3 accuracy is **75%** (at least three of four images
for each metric). Blur and multi-object scenes make this a useful small smoke
test, but these four images are neither independent of model selection nor
representative of production use. Passing is evidence against an obvious
regression, not proof of general accuracy or calibrated confidence.
