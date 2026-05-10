# DeepMIMO Random vs Contiguous

- num_seeds(random) = 3
- num_seeds(contiguous) = 3
- scale: K=4, Nt=8, Nsc=1

| Method | Random mean±std | Contiguous mean±std | Relative gap |
| --- | ---: | ---: | ---: |
| wmmse_iter_5 | 1.088417 ± 0.048405 | 1.066360 ± 0.000000 | -2.0265% |
| unfolded_wmmse_lite | 0.860601 ± 0.042589 | 0.871918 ± 0.000000 | +1.3149% |
| residual_rzf | 0.718788 ± 0.035922 | 0.763894 ± 0.000080 | +6.2753% |
| rzf | 0.714087 ± 0.035737 | 0.764191 ± 0.000000 | +7.0164% |
| cnn | 0.711433 ± 0.014816 | 0.657541 ± 0.027641 | -7.5751% |
