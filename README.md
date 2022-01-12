# Digital Image Processing Final Project

Reproducing [LRLG](https://arxiv.org/abs/1604.05817) and do some modification or improvement.

## Group Members

- [Yu-Kai Ling (r09921054)](r09921054@ntu.edu.tw) (Code Maintainer)
- [Po-Yen Tseng (r09521504)](r09521504@ntu.edu.tw) (Code Maintainer)
- [Li-Wei Fu (r10942078)](r10942078@ntu.edu.tw) 
- [Wei-Min Chu (r10546017)](r10546017@ntu.edu.tw)

## Usage

### Run in one line
```bash
bash run.sh <method> <name> <LR_output_dir> <repair_output_path>
```

* **method**: LR method (ex: LR/LRTV/LRL0/LRL0PHI)
* **name**: data name (ex: Teddy/Piano/Shelves/Vintage)
* **LR_output_dir**: path to LRxx output directory (ex: result/)
* **repair_output_path**: path to repaired output image (ex: result/repaired.png)

### Run step by step
1. Generate mask data

```bash
python gen_mask.py --input <input> --output_mask <output_mask> --output_missing <output_missing> --missing_rate <missing_rate>
```

* **input**: input depth image (ex: data/Teddy/disp.png)
* **output_mask**: path to mask output (ex: data/Teddy/mask.png)
* **output_missing**: path to depth missing output (ex: data/Teddy/missing.png)
* **missing_rate**: missing rate (ex: 0.5)

2. Run LR/LRTV/LRL0/LRL0Phi to repair the input data

```bash
python lrxx.py --method <method> --depth_image <depth_image> --mask <mask> [--init_image <init_image>] --output_path <output_path> --name <name> 
```

* **method**: LR method (ex: LR/LRTV/LRL0/LRL0PHI)
* **depth_image**: path to input depth image (ex: data/Teddy/disp.png)
* **mask**: path to input mask image (ex: data/Teddy/mask_50.png)
* **init_image**: path to input LR result image, LRTV/LRL0/LRL0PHI required (ex: data/Teddy/tnnr.png)
* **output_path**: path to result image (ex: result/)
* **name**: data name (ex: Teddy or Piano ...)

3. Run our method to fix the large holes in the result of LRL0Phi

```bash
python repair.py --input <input> --output <output>
```

* **input**: path to LR/LRTV/LRL0/LRL0Phi result image (ex: result/LRL0PHI_result/Teddy/lrl0phi.png)
* **output**: path to output repaired image (ex: repaired.png)

## Optional Usage

Visualizing the process of our method

```bash
python visualization --input <input> --output_dir <output_dir>
```

* **input**: path to repaired image (ex: repaired.png)
* **output**: path to output directory (ex: visualization/)
