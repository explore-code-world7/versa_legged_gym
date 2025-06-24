# algorithms:
1. 从env.step 获取数据;
2. process_env_step后处理计算奖励;
3. update alg's storatge with observations and policy's results
4. 优化 policy/critic network's weights


## ppo
* __init__():  create clip params
* init_storage(): create buffer dict, ppo contains RolloutStorage 
* act():    use actor_critic to infer action, values, and records them based on obs; 
* process_env_step():    add to self.transition()
* compute_returns():    compute discounted values; 
* update(): 
  1. generate batch-like dataset from self.storage.xxx_buffer;
  2. use batch data to update network;


# env
* vec_env: 定义虚拟环境的接口


# modules
## actor_cirtic
* __init__(): 定义各种超参
* xxx

## actor_critic_recurrent
* rnn接受[num_transitions_per_env, trajectory_length, observation_dim]

# networks
* components of policy&critic network
## memory
* lstm to pre-process observations into hidden_states;
* reset(): 
* forward(): feed-forward observations and states 
* detach_hidden_states():


# runners
* 每个runner绑定env, alg, module(with networks)
## on_policy_runner
* learn():    in each step, act all envs to get buffer, use update to train network;
* log(): 
* save():    
* load(): 
* get_inferece_policy();


# storage
* rollout_storage和replay_buffer的区别是什么?
* replay_buffers仅仅获取观察，rollout_storage从观察中，构建

## rollout_storage
* __init__(): init self.buffers
* add_transitions: update_buffer with data from class transition
```python
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
        
        def clear(self):
            self.__init__()
```
   * _save_hidden_states: save hidden states;
* compute_returns: computes generalized advantage estimation;
* get_statistics(): returns trajectory and rewards;
* mini_batch_generator: produce batch-like dataset from self.buffer
* recurrent_mini_batch_generator:

## replay_buffer
* 维护一个states_buffer with size = `(self.buffer_size, obs_dim)`, 并从中采样
* __init__: initialize states, next_states
* insert: insert a batch of states and next_states into states_buffer
* feed_forward_generator: yield num=num_mini_batch * mini_batch_size states for build dataset


# utils
## utils
* 定义一些和库相关的函数
* resolve_nn_activation: return activation function
* split_and_pad_trajectories: 拆分轨迹
* 将[num_transtions_per_env, num_env, obs_dim]拆分为[num_trans_per_env, num_trajectory_num, obs_dim]
* 一个env在一次num_transtions_per_env可生成若干trajectory;


*** 
# Parkour
* runner_builder的作用是什么?


# util
## buffer
* buffer_from_example: 把example放在share_memory中，并修改维度为leading_dim


## ckpt_manipulator
* 根据给定的源状态字典（source_state_dict）和算法状态字典（algo_state_dict）来替换模型的编码器权重

## collections
* nametuple, namearraytuple的构造、使用各种方法;
* namedarraytuple: 
* namedtuple: 
* namedtupleschema:

### namedtuple usage brochure
```python
# 把namedarraytuple转为tuple
if is_namedarraytuple(self.hidden_states):
    self.hidden_states = tuple(self.hidden_states)
out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
# 得到结果tuple(tensor, tensor)
if isinstance(self.hidden_states, tuple):
    self.hidden_states = LstmHiddenState(*self.hidden_states)
out = out.squeeze(0) # remove the time dimension
```

## data_compresser
* compress_normalized_image: compress image from np.float [0,1] to np.uint8

# Storage
## rollout_storage
* observations.shape: [num_transitions_per_env, num_envs, obs_shape]
* _save_hidden_states:  将hidden_states作为一个numpy

* get_minibatch_from_indices: 
  1. 根据给定的T_slices, B_slices输出obs_batch
  2. 如果由padded_B_slice, 输出hidden_batch和obs_mask_batch

# Modules
## actor_critic
## actor_critic_recurrent
* LstmHiddenState, ActorCriticHiddenState 定义lstm层和actor_critic hidden层上;
* Memory: process hidden_states


# Algorithms
## ppo
* compute_losses:
  1. actor_critic.act的输入hidden_states 采用自定义结构

* state_dict: 输出算法的model/optimizer/lr_scheduler_state_dict，用于保存\
```python
# ppo_runner示例
    def save(self, path, infos=None):
        run_state_dict = self.alg.state_dict()
        run_state_dict.update({
            'iter': self.current_learning_iteration,
            'infos': infos,
        })
        torch.save(run_state_dict, path)
```
* load_state_dict: 从预训练state_dict加载网络

# Runner
* None