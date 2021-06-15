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
env = gym.make('msgr-train-v2')
obs, manual = env.reset()
obs, reward, done, info = env.step(<some action>)
```
Here `<some action>` should be an integer between 0 and 4 corresponding to the actions `up`,`down`,`left`,`right`,`stay`. Notice that in contrast to standard gym, `env.reset()` returns a tuple of an observation and the text manual sampled for the current episode. If you have installed the model dependencies, you can use our model `EMMA` just as you would any `torch` model. A full example of using `EMMA` with our environment can be found in the `run.py` file. You can run it with:
```
python run.py --model_state pretrained/s2_sm/sm_s2_1_max.pth --env_id msgr-train-v2
```

Pretrained model weights are available in the `pretrained` folder. Please make sure that you load the correct weights for the correct environment stages. (`v1` environments should use model states in the `pretrained/s1_sm` folder and `v2` environments from the `pretrained/s2_sm` folder).

### Environment IDs
Environment ids follow the following format: `msgr-{split}-v{stage}`. There are three stages (1,2,3) and the splits include: `train`, `val`, `test`, as well as `train-sc` and `train-mc` for the single and multi-combination subsets of the training games. The split `test-se` is the state estimation version of the test environment, and is only available on stage 2. Note that compared we have introduced a stage 3, and `msgr-test-v2` includes more movement combinations for a more comprehensive test. Other stages/splits should be identical.

## Miscellaneous
If there are issues with the installation, try using `Python 3.7`. The model is tested working with `transformers` version 4.2.2.
