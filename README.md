# gym-messenger
Gym implementation of the Messenger environment.

## Installation
Currently, only local installations are supported. Clone the repository and run:
```
pip install -e gym-messenger
```
This will install only the Messenger gym environments and dependencies. If you want to use models, you must install additional dependencies such as `torch`. Run the following instead:
```
pip install -e 'gym-messenger[models]'
```

## Usage
To instantiate a gym environment, use:
```python
import gym
env = gym.make('msgr-train-v2')
obs, manual = env.reset()
obs, reward, done, info = env.step(<some action>)
```
Notice that in contrast to standard gym, `env.reset()` returns a tuple of an observation and the text manual. If you have installed the model dependencies, you can use our model `EMMA` just as you would any `torch` model. A full example of using `EMMA` with our environment can be found in the `run.py` file. You can run it with:
```
python run.py --model_state pretrained/s2_sm/sm_s2_1_max.pth --env_id msgr-train-v2
```

Pretrained model weights are available in the `pretrained` folder. Please make sure that you load the correct weights for the correct environment stages. (`v1` environments should use model states in the `pretrained/s1_sm` folder and `v2` environments from the `pretrained/s2_sm` folder).

### Environment IDs
The following environment ids are currently available:
- `msgr-train-v1`
- `msgr-train-sc-v1`
- `msgr-train-mc-v1`
- `msgr-val-v1`
- `msgr-test-v1`
- `msgr-train-v2`
- `msgr-train-sc-v2`
- `msgr-train-mc-v2`
- `msgr-val-v2`
- `msgr-test-v2`
- `msgr-test-se-v2`

`v1, v2` represent stages 1 and 2 respectively, `sc, mc` are the single and multi-combination training games respectively. `train` samples from both `sc` and `mc` games. `val` are the validation games and features the same movement dynamics as those in `train`. `test` games feature new movement combinations. `test-se` is the state-estimation version of the test games that feature the same movement dynamics as those in `train`. For more details refer to the [paper](https://arxiv.org/abs/2101.07393).

## Miscellaneous
If there are issues with the installation, try using `Python 3.7` (later versions may cause problems). Note that only tools for running the models are provided, there are no tools to train the models (yet). Negation and neutral entities have also not been implemented.
