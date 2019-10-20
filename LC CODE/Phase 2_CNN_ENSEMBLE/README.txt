The code to the solution is located here.

The lidc-idri, luna16 (subset of lidc-irdi), and kaggle national data science bowl 2017 datasets are necessary to train the ensemble.

The files should be run in this sequence to train the network:
1.preprocess_cubes.py
2.anom_mass_segmenter.py
3.nodule_detector.py
4.predict_nodules.py
5.final_linear_mod.py

To predict nodules, the predict_nodules.py file may be run alone.

I left out a few messy, initial jupyter notebooks and making of figures. I tried to clean up the code a little bit; however, there are still some pretty messy, and nonintuitive portions. File paths are mostly explicit and need to be changed.
