# PNN
## An Implementation Of Progressive Neural Networks for Intrusion Detection Systems <br/>
This repository hosts a basic implementation of Progressive Neural Networks(PNN) as stated in the original paper (https://arxiv.org/pdf/1606.04671.pdf).<br/>
This implementation will be used in a research project conducted in the Information, Network, and Learning Lab ([INL](https://inl-lab.net/)) of Sharif University of Technology. After the research paper is published, further experiments and results will be provided. <br/>
Going through the code, you might notice that a data.py file is missing. This file contains a controller class which function is to generate train, validation, and test data (with respect to the configured batch size) from the vectorized flows obtained from the [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset.<br/>
Since the data used for the experiments is exclusive to the INL, this file can't be shared before the paper gets published. Nevertheless, implementing a controller isn't tricky, so you can write your own code for the controller based on your data and replace it in the code.
