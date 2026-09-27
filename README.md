

# How to run:
There are 3 make files. The **Main** Makefile, **Data** Makefile and **Report** Makefile.

The **Main** makefile doesnt do any setup. Setup requires running setup in each of the **Data** and **Report** Makefile.

Running test generation take ~15 minutes per run on lab PC. It runs endlessly untill it is stoped unless told otherwise.


## Setup:

Enter the **Data** dir.

```
cd data
```

Run setup.
```
make setup
```

<details>
<summary>Report does not require setup.</summary>
The setup for the report is generating the plots which is done each time the normal pdf generation is run.
</details>


## Running:

### Running data generation:

To run data generation in **MAIN** Dir:
```
make runData
```

To run data generation in **DATA** Dir:
```
make run
```
Options:

```
make run ARGS="<--skip-baseline,--skip-incremental,--runs={number of runs}>"

```

To generate plots: (This is done automaticly by report)
```
make plots
```


### Running report:


To run report generation in **MAIN** Dir:
```
make runReport
```

To run report generation in **Report** Dir:
```
make compilePDF
```


# Report:


[Open report](https://hannauwlaing.github.io/MachineLearning741Assignment3/report.pdf)

### Report preview

![Report page 1](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-1.png)
![Report page 2](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-2.png)
![Report page 3](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-3.png)
![Report page 4](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-4.png)
![Report page 5](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-5.png)
![Report page 6](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-6.png)
![Report page 7](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-7.png)
![Report page 8](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-8.png)
![Report page 9](https://hannauwlaing.github.io/MachineLearning741Assignment3/report-pages/page-9.png)
