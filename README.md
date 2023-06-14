# Messenger-EMMA
Implementation of the Messenger environment and EMMA model from the ICML 2021 paper: [Grounding Language to Entities and Dynamics for Generalization in Reinforcement Learning](https://arxiv.org/abs/2101.07393). 

## Installation
Currently, only local installations are supported. Clone the repository and run:
```
pip install -e messenger-emma
```
This will install only the Messenger gym environments and dependencies. If you want to use models, you must install additional dependencies such as `torch`. Run the following instead:
```
pip install -e 'messenger-emma[models]'
```

## Usage
To instantiate a gym environment, use:
```python
import gym
import messenger
env = gym.make('msgr-train-v2')
obs, manual = env.reset()
obs, reward, done, info = env.step(<some action>)
```
Here `<some action>` should be an integer between 0 and 4 corresponding to the actions `up`,`down`,`left`,`right`,`stay`. Notice that in contrast to standard gym, `env.reset()` returns a tuple of an observation and the text manual sampled for the current episode. If you have installed the model dependencies, you can use our model `EMMA` just as you would any `torch` model. A full example of using `EMMA` with our environment can be found in the `run.py` file. To download model weights run the following:
```
wget -O pretrained.zip https://www.dropbox.com/s/ne8yglb0765f111/pretrained.zip?raw=1
unzip pretrained.zip
```
This will put pretrained model weights in a folder called `pretrained`. You can run EMMA using these weights with:
```
python run.py --model_state pretrained/emma_s2_1.pth --env_id msgr-train-v2
```
 Please make sure that you load the correct weights for the correct environment stages. `v1`, `v2`, `v3` environments should use model states with `s1`, `s2`, `s3` in the filename respectively.

 ### Training

Training scripts and usage information is provided in the folder [training](./training/)

### Environment IDs
Environment ids follow the following format: `msgr-{split}-v{stage}`. There are three stages (1,2,3) and the splits include: `train`, `val`, `test`, as well as `train-sc` and `train-mc` for the single and multi-combination subsets of the training games. The split `test-se` is the state estimation version of the test environment, and is only available on stage 2. Please ignore any warnings from `gym` telling you to "upgrade to v3".

### Human Play
To get a better sense of what Messenger is like, you can play it in the terminal assuming you have installed the environment. Specify the `--env_id` to the gym id you want to play:
```
python play_msgr.py --env_id msgr-train-v1
```
Note that in this human-play version, the entity groundings are provided to you upfront by rendering each entity with its first two letters (e.g. airplane as `AI`). In the actual environment, the agent must learn this grounding from scratch by matching text symbols like "plane" to the symbol `2`.

## Environment Details
This section documents some of the nuances of the environment and its usage.

### Additional Hyperparameters

On stage 1, the agent begins with or without the message and wins the episode if it interacts with the correct entity. As a default, the agent begins with the message with prob 0.2 at the start of each episode. You can change this parameter as follows (note that this only applies to stage 1.):
```python
env = gym.make("msgr-train-v1", message_prob=0.5)
```

On training games, there is a concept of single and multi-combination games. Since there are not many single-combination game variants, we sample one of these games with probability 0.25. the `prob_env_1` keyword sets the probability of sampling a multi-combination game (which is 0.75 by default). You can change this with:
```python
env = gym.make("msgr-train-v1", prob_env_1=0.6)
```
Note that there are no concepts of single and multi-combination games on test or validation games.

### Step Limits and Penalities

The gym environment does not implemenet any sort of step limit or step penalty. This is to allow for maximum flexibility for various training setups (for example, you might want to start with a higher limit, and then anneal it over the course of training). Note that since entities are not always chasing, depending on the quality of the agent, some episodes may never terminate if no limit is specified, so we recommend including one in your training loop. During training, we also penalized the agent with a -1 reward if it did not complete the episode within our step limit.

### Text Manual

Due to the noisy nature of data collected from human writers, sometimes the manual may contain a description that provides no useful information. In most cases, the correct course of action can still be deduced by reading the other descriptions.

## Changes
- July 07 2021: Added script for playing Messenger in the terminal. Removed pretrained weights from repo and moved it to Dropbox.
- June 15 2021: We have introduced a stage 3, and `msgr-test-v2` which includes more movement combinations for a more comprehensive test. Other stages/splits should be identical. If you cloned before this `8f6bd5c` commit, we recommend getting the latest version.

## Miscellaneous
If there are issues with the installation, try using `Python 3.7`. The model is tested working with `transformers` version `4.2.2`. The license is MIT. If you get an error with `gym` try downgrading `gym` to `0.22.0` or lower.

Please use the following citation from [DBLP](https://dblp.org/rec/conf/icml/HanjieZN21.html?view=bibtex) (note author list and name changes from early arxiv versions).

```
@inproceedings{hanjie21grounding,
  author    = {Austin W. Hanjie and
               Victor Zhong and
               Karthik Narasimhan},
  editor    = {Marina Meila and
               Tong Zhang},
  title     = {Grounding Language to Entities and Dynamics for Generalization in
               Reinforcement Learning},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning,
               {ICML} 2021, 18-24 July 2021, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {139},
  pages     = {4051--4062},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {http://proceedings.mlr.press/v139/hanjie21a.html},
  timestamp = {Wed, 14 Jul 2021 15:41:58 +0200},
  biburl    = {https://dblp.org/rec/conf/icml/HanjieZN21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
