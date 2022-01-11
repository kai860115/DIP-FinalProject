# Digital Image Processing Final Project

Reproducing [LRLG](https://arxiv.org/abs/1604.05817) and do some modification or improvement.

## Group Members

- [Yu-Kai Ling (r09921054)](r09921054@ntu.edu.tw) (Code Maintainer)
- [Po-Yen Tseng (r09521504)](r09521504@ntu.edu.tw) (Code Maintainer)
- [Li-Wei Fu (r10942078)](r10942078@ntu.edu.tw) 
- [Wei-Min Chu (r10546017)](r10546017@ntu.edu.tw)

## Usage

Generate input data

```bash
python ?????
```

Run LRL0Phi to repair the input data

```bash
python lrxx.py --method LRL0PHI
```

Run our method to fix the large holes in the result of LRL0Phi

```bash
python repair.py
```

## Optional Usage

Using LRTV to repair the input data

```bash
python lrxx.py --method LRTV
```

Using LRTV to repair the input data

```bash
python lrxx.py --method LRL0
```

Visualizing the process of our method

```bash
python visualization
```