## Releases

PDF releases to reviewers or conferences are collected here. Also the camera-ready version with copyright disclaimer for open access publication will be found here.


### File namings

It's good practice to add a `CONFIDENTIAL_*` to PDFs that are not publically released yet, e.g.
* `CONFIDENTIAL_kuhnt_topic2016_v0.5.pdf`
* `CONFIDENTIAL_kuhnt_topic2016_v1.0_cam_ready.pdf`

and only remove the `CONFIDENTIAL_*` tag at PDFs that can be released publically:
* `kuhnt_topic2016_copyright.pdf`


### Open Access PDF

To create the final version with copyright disclaimer for open access (e.g. researchgate), you have to:

1. update the `copyright.tex` in the main folder with your information
2. build the `copyright.tex` (to `copyright.pdf`)
3. update the `add_copyright.bash` script to your file namings
4. run the script
