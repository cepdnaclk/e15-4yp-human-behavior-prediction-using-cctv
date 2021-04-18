# Adaptive people movement and action prediction using CCTV to control appliances

#### Table of content


1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
5. [Results and Analysis](#results-and-analysis)
6. [Conclusion](#conclusion)
7. [Publications](#publications)
8. [Links](#links)


## Introduction

##Abstract

With the availability of high-performance processors
and GPUs, the demand for Machine learning, Deep learning
algorithms is growing exponentially. It has become more and
more possible to explore the depths of fields like Computer
vision with these trends. Detecting humans in video footage using
computer vision is one such area. Although human detection is
somewhat primitive when compared to today’s technology, using
that data to produce various results like recognizing postures,
predicting behaviors, predicting paths are very advanced fields
and they have very much room left to grow. Various algorithms,
approaches are available today to accomplish the above kind of
tasks, from classical machine learning, neural networks to statistical
approaches like Bayes theorem, Hidden Markov Models,
Time series, etc. This paper summarize the result of a system
that combines above technologies in order to control electric
appliances through predictions. These predictions are deducted
by analyzing CCTV footages of the user using computer vision.


## Requirments

## Installation
First clone the repository

```bash
git clone https://github.com/RisithPerera/e15-4yp-human-behavior-prediction-using-cctv.git
cd e15-4yp-human-behavior-prediction-using-cctv/code
```
#### Download pretrained weights

```bash
./weights/download_weights.sh
```

## How to Run

```bash
python CCTVObserver.py
python FaceObserver.py
```

## Results
