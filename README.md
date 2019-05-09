Sequential Specialization
-------------------------
In this project, we observe that day-to-day video exhibits highly skewed class distributions over the short intervals. We demonstrate that when class distribution is highly skewed toward small sets of classes, “specialized” CNNs trained to classify inputs from this distribution can be much simpler than the oracle classifier. We formulate the problem of detecting the short-term skews online and exploiting models based on it as a new sequential decision making problem dubbed the Online Bandit Problem, and present a new algorithm to solve it. When applied to recognizing faces in TV shows and movies, we realize end-to-end classification speedups of 2.4-7.8x/2.6-11.2x (on GPU/CPU) relative to a state-of-the-art convolutional neural network, at competitive accuracy.

## Pre-requisite
- Install caffe [pre-requisite](https://caffe.berkeleyvision.org/installation.html#prerequisites) with GPU support
- Install python package: `numpy cv2 scikit-image plyvel`
- Currently only compatible with Python 2.7.

## Instructions
1. Compile Caffe and set the python environment to caffe python.
2. Compile the protobuf by running `make proto`.
3. Download pretrained model by running `download_pretrain_model.sh`.
4. Config the dataset path in `env.cfg`.
5. Run the test script
    ```
    python test_runtime.py --type video --task face --model F1 -i video_traces/departed_trace.txt -o output.txt
    ```

## Publication
[Fast Video Classification via Adaptive Cascading of Deep Models](https://syslab.cs.washington.edu/papers/specialization-cvpr17.pdf)\
Haichen Shen, Seungyeop Han, Matthai Philipose, and Arvind Krishnamurthy\
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), July 2017.