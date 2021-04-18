# Adaptive people movement and action prediction using CCTV to control appliances

#### Table of content

1. [Introduction](#intro)
2. [Abstract](#abstract)
3. [Requirments](#requirments)
4. [Installation](#installation)
5. [How to Run](#how-to-run)
6. [Results](#results)

## Introduction

Throughout this project, two main aspects will be concerned.
The first one is tracking and predicting people’s
movements and the second one is recognizing and predicting
people’s actions. Since these two aspects are already taken
the interest of many researchers of the world, there are many
different approaches with different performance available for
the above tasks. And many of these approaches are not
independent of each other as many common libraries and
frameworks are used to develop them. Therefore combining
a various number of such components while keeping the
overhead to the CPU & GPU at a minimum is a complex task.
Therefore this paper will report the results of this project which
is an attempt to achieve the above-mentioned two aspects
while making the result be less resource-intensive as possible
so that the final system can be implemented even on a personal
computer.
This paper will describe the project using the data flow
within the system. For clarity, the flow of data will be
breakdown into three subdomains as human identification,
behavior extraction, and behavior prediction.
In the human identification phase, a YOLO v3 model is
used to detect humans entering the premises at the entrance.
Once a human is detected the cropped image will be used as
the input to an MTCNN model that is used to detect the face
of the human. Afterward, the result is given as an input to an
Inception ResNet V1 model to extract the features of the face.
In the behavior extraction phase, a YOLO v3 model is used
again to detect humans in the frame. Then OpenPose is used to
draw the skeletons of each person within the bounding boxes
drawn using the YOLO model. Once the skeletons are drawn,
those key points are sent to a classifier to recognize the action.
Then this data is recorded in the database.
In the behavior prediction phase, a Hidden Markov Model
is used to detect the next possible location. Then the current
data of the person is sent to a CatBoost classifier along with
the result from the HMM model. The classifier determines
whether it is required to turn a specific appliance on or off.
The Figure 1 gives a more elaborated explanation of the
data flow in the system.

## Abstract

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
