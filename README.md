# Noisy Channel And N-gram Model Based Spelling Correction

See the detailed project description in the **"spelling_project.pdf"**.

## Project files:
|                      |                                           |
|----------------------|-------------------------------------------|
| SP_main.py 		       | main script with the implemented model    |
| SP_func.py 		       | script with functions used in SP_main.py  |
| spell.py 		         | P.Norvigs off-the-shelf model             |
| testsets.py 		     | test items for evaluation                 |
| big.txt		           | language corpus                           |
| missp.dat		         | error corpus                              |
| spelling_project.pdf |	project report                           | 

## Instructions:

1. Run SP_main.py to produce the output with test results for the implemented model.
2. Run spell.py to test the off-the-shelf model.

Expected runtime of SP_main.py: 45 min (appx. 5 minutes per each test); can be reduced by tuning parameters of the "test" function in SP_main.py

## References
Jurafsky, D., & Martin, J. H. (2021). Appendix B: Spelling Correction and the Noisy Channel. In Speech and Language Processing. https://web.stanford.edu/~jurafsky/slp3/

Norvig, P. (2007). How to Write a Spelling Corrector. http://norvig.com/spell-correct.html

