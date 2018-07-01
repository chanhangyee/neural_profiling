
# Analysis pipeline

_Refer to the original article for page numbers_

## Preprocessing

- Realignment and co-registration
- 3mm FWHM smoothing (refer to `p.53`)

## Preparing neural context scores and neural profiles

- Obtain contrast maps of each social context pair comparison (`p.14`): `spm\Create Picture Category Betas.py`
- Obtain ROI mask based on contrast maps (`p.14`): `Preprocessing 1 - Create ROI masks.ipynb`
- Extract raw voxel data using the mask (`p.14`): `Preprocessing 2 - Extract voxel data.ipynb`
- Calculate neural context scores and neural profiles (`p.15-16`): `Preprocessing 3 - Calculate scores.ipynb`

## Main analysis

- The average classification accuracy is 44.9%, SD = 8.2%, which is significantly above chance at 25% (t(36) = 14.6, p < .0001), indicating that the voxels contained information for social context decoding. (`p.16`): `Analysis 1 - Picture category accuracy.ipynb`
- Overall, neural context scores significantly correlated with survey responses (F(1,1501.28) = 15.7, p < .0001) (`p.17`): `Analysis 2a - Brand perception.ipynb` and `Analysis 2b - Brand perception.ipynb` (mixed-effect modelling in R)
- The average correlation (after Fisher’s r-to-z transformation, Silver & Dunlap, 1987) was .107, and the Fisher-transformed correlations were significantly different from zero (t(36) = 6.16, p < .0001). (`p.18`): `Analysis 3 - Brand dissimilarity.ipynb`
- We found the relationship between the aggregated interbrand neural profile disparity matrix and the co-branding suitability matrix to be significantly negative (Figure 6B, r = -.384, p < .0001). (`p.22`): `Analysis 3 - Brand dissimilarity.ipynb`
- However, the correlation between average intersubject neural profile disparity and brand image strength was significant (Figure 7, r = -.627, p = .013 based on permutations) (`p.23`): `Analysis 4 - Brand image strength.ipynb`

## Supplementary analysis

- `Analysis 1 (Robustness check).ipynb`
- `Analysis 2 (Robustness check).ipynb`
- `Analysis 3 (Robustness check).ipynb`
