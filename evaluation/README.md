# Evaluation dataset

Four fixed scikit-image v0.25.2 photographs. The
[manifest](manifest.json) records accepted ImageNet labels, source URLs,
authors, redistribution permissions, and SHA-256 checksums.

| Image | Author | Permission | Accepted labels |
| --- | --- | --- | --- |
| Chelsea | Stefan van der Walt | CC0 | tabby, tiger cat, Egyptian cat |
| Coffee | Rachel Michetti, courtesy of Pikolo Espresso Bar | CC0 | espresso |
| Clock | Stefan van der Walt | Public domain | analog clock, wall clock |
| Camera | Lav Varshney | CC0 | tripod, reflex camera |

Permissions are documented in
[scikit-image's data reference](https://scikit-image.org/docs/0.25.x/api/skimage.data.html).
[CC0](https://creativecommons.org/publicdomain/zero/1.0/) permits redistribution
and modification. The camera fixture is the CC0 replacement introduced in
scikit-image 0.18.

Multiple labels accommodate overlapping domestic-cat categories or multiple
visible objects. Each image counts once. The manifest records the label rationale.

Evaluation requires **75% top-1** and **75% top-3** accuracy: at least three of
four samples for each metric. The [recorded baseline](baseline.json) reaches
75% and 100%, respectively. The blurred clock is classified as `bubble` first
and `wall clock` third.

This collection is a regression smoke test. Its small size, limited coverage,
and use during development make it unsuitable for estimating deployment accuracy
or calibrating confidence scores.
