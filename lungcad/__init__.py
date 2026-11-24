"""lungcad — thesis code for hybrid deep-learning lung-nodule detection (WP0 → WP4).

Package layout:
  lungcad.config        paths + the constants every stage shares (HU window, spacings)
  lungcad.preprocess    S2: the ONE shared preprocessing module (LUNA16 + NLST)
  lungcad.augment       dose / kernel augmentation (S4)
  lungcad.data          S1 index, LUNA16 preprocessing driver, NLST inventory/conversion/labels
  lungcad.stage1        RetinaNet 3D detector: model, dataset, train, infer (S3, S5)
  lungcad.stage2        FP-reduction models A/B/C: models, patches, dataset, train, infer (S6)
  lungcad.eval          official-LUNA16-equivalent FROC/CPM scorer + paired bootstrap
  lungcad.wp4           patient-level aggregators (max-pool, feature GBM, attention-MIL)
"""
__version__ = "0.1.0"
