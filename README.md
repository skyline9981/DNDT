# DNDT

## requirements

- python3
- numpy
- matplotlib
- sklearn
- torch
- pandas
- functools

## usage

```bash
python3 DNDT.py
python3 DT.py
```

## Warning

- **The hyperparameters are not tuned, so the results may be different from the paper.**
- **The seed is not fixed, so you may get different results each time you run the code.**

## Dataset Description

### Connect-4

- [UCI](https://archive.ics.uci.edu/ml/datasets/Connect-4)

1. Title: Connect-4 opening database

2. Source Information
   (a) Original owners of database: John Tromp (tromp@cwi.nl)
   (b) Donor of database: John Tromp (tromp@cwi.nl)
   (c) Date received: February 4, 1995

3. Past Usage: not available

4. Relevant Information:
   This database contains all legal 8-ply positions in the game of
   connect-4 in which neither player has won yet, and in which the next
   move is not forced.

5. Number of Instances: 67557

6. Number of Attributes: 42, each corresponding to one connect-4 square

7. Attribute Information: (x=player x has taken, o=player o has taken, b=blank)

   The board is numbered like

   |     | a   | b   | c   | d   | e   | f   | g   |
   | --- | --- | --- | --- | --- | --- | --- | --- |
   | 6   | .   | .   | .   | .   | .   | .   | .   |
   | 5   | .   | .   | .   | .   | .   | .   | .   |
   | 4   | .   | .   | .   | .   | .   | .   | .   |
   | 3   | .   | .   | .   | .   | .   | .   | .   |
   | 2   | .   | .   | .   | .   | .   | .   | .   |
   | 1   | .   | .   | .   | .   | .   | .   | .   |

   1. a1: {x,o,b}
   2. a2: {x,o,b}
   3. a3: {x,o,b}
   4. a4: {x,o,b}
   5. a5: {x,o,b}
   6. a6: {x,o,b}
   7. b1: {x,o,b}
   8. b2: {x,o,b}
   9. b3: {x,o,b}
   10. b4: {x,o,b}
   11. b5: {x,o,b}
   12. b6: {x,o,b}
   13. c1: {x,o,b}
   14. c2: {x,o,b}
   15. c3: {x,o,b}
   16. c4: {x,o,b}
   17. c5: {x,o,b}
   18. c6: {x,o,b}
   19. d1: {x,o,b}
   20. d2: {x,o,b}
   21. d3: {x,o,b}
   22. d4: {x,o,b}
   23. d5: {x,o,b}
   24. d6: {x,o,b}
   25. e1: {x,o,b}
   26. e2: {x,o,b}
   27. e3: {x,o,b}
   28. e4: {x,o,b}
   29. e5: {x,o,b}
   30. e6: {x,o,b}
   31. f1: {x,o,b}
   32. f2: {x,o,b}
   33. f3: {x,o,b}
   34. f4: {x,o,b}
   35. f5: {x,o,b}
   36. f6: {x,o,b}
   37. g1: {x,o,b}
   38. g2: {x,o,b}
   39. g3: {x,o,b}
   40. g4: {x,o,b}
   41. g5: {x,o,b}
   42. g6: {x,o,b}
   43. Class: {win,loss,draw}

### German Credit Data

- [UCI](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29)

1. Title: German Credit data

2. Source Information

   Professor Dr. Hans Hofmann  
   Institut f"ur Statistik und "Okonometrie  
   Universit"at Hamburg  
   FB Wirtschaftswissenschaften  
   Von-Melle-Park 5  
   2000 Hamburg 13

3. Number of Instances: 1000

   Two datasets are provided. the original dataset, in the form provided
   by Prof. Hofmann, contains categorical/symbolic attributes and
   is in the file "german.data".

   For algorithms that need numerical attributes, Strathclyde University
   produced the file "german.data-numeric". This file has been edited
   and several indicator variables added to make it suitable for
   algorithms which cannot cope with categorical variables. Several
   attributes that are ordered categorical (such as attribute 17) have
   been coded as integer. This was the form used by StatLog.

4. Number of Attributes german: 20 (7 numerical, 13 categorical)
   Number of Attributes german.numer: 24 (24 numerical)

5. Attribute description for german

   - Attribute 1: (qualitative)
     Status of existing checking account
     A11 : ... < 0 DM
     A12 : 0 <= ... < 200 DM
     A13 : ... >= 200 DM /
     salary assignments for at least 1 year
     A14 : no checking account

   - Attribute 2: (numerical)
     Duration in month

   - Attribute 3: (qualitative)
     Credit history
     A30 : no credits taken/
     all credits paid back duly
     A31 : all credits at this bank paid back duly
     A32 : existing credits paid back duly till now
     A33 : delay in paying off in the past
     A34 : critical account/
     other credits existing (not at this bank)

   - Attribute 4: (qualitative)
     Purpose
     A40 : car (new)
     A41 : car (used)
     A42 : furniture/equipment
     A43 : radio/television
     A44 : domestic appliances
     A45 : repairs
     A46 : education
     A47 : (vacation - does not exist?)
     A48 : retraining
     A49 : business
     A410 : others

   - Attribute 5: (numerical)
     Credit amount

   - Attibute 6: (qualitative)
     Savings account/bonds
     A61 : ... < 100 DM
     A62 : 100 <= ... < 500 DM
     A63 : 500 <= ... < 1000 DM
     A64 : .. >= 1000 DM
     A65 : unknown/ no savings account

   - Attribute 7: (qualitative)
     Present employment since
     A71 : unemployed
     A72 : ... < 1 year
     A73 : 1 <= ... < 4 years  
      A74 : 4 <= ... < 7 years
     A75 : .. >= 7 years

   - Attribute 8: (numerical)
     Installment rate in percentage of disposable income

   - Attribute 9: (qualitative)
     Personal status and sex
     A91 : male : divorced/separated
     A92 : female : divorced/separated/married
     A93 : male : single
     A94 : male : married/widowed
     A95 : female : single

   - Attribute 10: (qualitative)
     Other debtors / guarantors
     A101 : none
     A102 : co-applicant
     A103 : guarantor

   - Attribute 11: (numerical)
     Present residence since

   - Attribute 12: (qualitative)
     Property
     A121 : real estate
     A122 : if not A121 : building society savings agreement/
     life insurance
     A123 : if not A121/A122 : car or other, not in attribute 6
     A124 : unknown / no property

   - Attribute 13: (numerical)
     Age in years

   - Attribute 14: (qualitative)
     Other installment plans
     A141 : bank
     A142 : stores
     A143 : none

   - Attribute 15: (qualitative)
     Housing
     A151 : rent
     A152 : own
     A153 : for free

   - Attribute 16: (numerical)
     Number of existing credits at this bank

   - Attribute 17: (qualitative)
     Job
     A171 : unemployed/ unskilled - non-resident
     A172 : unskilled - resident
     A173 : skilled employee / official
     A174 : management/ self-employed/
     highly qualified employee/ officer

   - Attribute 18: (numerical)
     Number of people being liable to provide maintenance for

   - Attribute 19: (qualitative)
     Telephone
     A191 : none
     A192 : yes, registered under the customers name

   - Attribute 20: (qualitative)
     foreign worker
     A201 : yes
     A202 : no
