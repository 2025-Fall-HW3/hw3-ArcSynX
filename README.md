[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/ZHXGkqs_)
# 2025-Spring-HW3

## Installation

1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. deactivate // leave the virtual environment

## Part 1

### Assignment Score

Problem 1 - equal weight portfolio

`python Markowitz.py --score eqw`

Problem 2 - risk parity portfolio

`python Markowitz.py --score rp`

Problem 3 - mean variance portfolio

`python Markowitz.py --score mv`

All Problems

`python Markowitz.py --score all`

### Asset Allocation

Problem 1 - equal weight portfolio

`python Markowitz.py --allocation eqw`

Problem 2 - risk parity portfolio

`python Markowitz.py --allocation rp`

Problem 3 - mean variance portfolio

`python Markowitz.py --allocation mv`

### Performance Check

Problem 3 - mean variance portfolio

`python Markowitz.py --performance mv`

### Evaluation Metric

Problem 3 - mean variance portfolio

`python Markowitz.py --report mv`

## Part 2

### Assignment Score

Problem 1 - Sharp Ratio > 1

`python Markowitz_2.py --score one`

Problem 2 - Sharp Ratio > SPY

`python Markowitz_2.py --score spy`

All Problems

`python Markowitz_2.py --score all`

### Asset Allocation

My portfolio

`python Markowitz_2.py --allocation mp`

Big portfolio

`python Markowitz_2.py --allocation bmp`

### Performance Check

My portfolio

`python Markowitz_2.py --performance mp`

Big portfolio

`python Markowitz_2.py --performance bmp`

### Evaluation Metric

My portfolio

`python Markowitz_2.py --report mp`

Big portfolio

`python Markowitz_2.py --report bmp`

### Cumulative Result

My portfolio

`python Markowitz_2.py --cumulative mp`

Big portfolio

`python Markowitz_2.py --cumulative bmp`


