# Neural Differential Equations for Learning to Program Neural Nets (Continuous Fast Weight Programmers)

This is the official repository containing code for the paper:

[Neural Differential Equations for Learning to Program Neural Nets Through Continuous Learning Rules (NeurIPS 2022)](https://arxiv.org/abs/2206.01649)


## Contents

* `speech_and_physionet` directory contains the code used for the Speech Commands and PhysioNet Sepsis experiments (Table 1). Originally forked from [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE).
* `eigenworms` directory contains the code used for the EigenWorms experiment (Table 2). Originally forked from [jambo6/neuralRDEs](https://github.com/jambo6/neuralRDEs).
* `appendix_mujoco` directory for the extra reinforcement learning experiments presented in the appendix. Originally forked from [dtak/mbrl-smdp-ode](https://github.com/dtak/mbrl-smdp-ode).

Separate license files can be found in each of these directories.

## Links
* Models implemented here are the continuous-time counterparts of Fast Weight Programmers. For the discrete-time models, see our previous works: 
    * [Linear Transformers are Secretly Fast Weight Programmers (ICML 2021)](https://arxiv.org/abs/2102.11174)
    * [Going Beyond Linear Transformers with Recurrent Fast Weight Programmers (NeurIPS 2021)](https://arxiv.org/abs/2106.06295)
    * [A Modern Self-Referential Weight Matrix That Learns to Modify Itself (ICML 2022)](https://arxiv.org/abs/2202.05780)
* [Jürgen Schmidhuber's AI blog post on Fast Weight Programmers (March 26, 2021)](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html).

## BibTex
```
@inproceedings{irie2022neural,
  title={Neural Differential Equations for Learning to Program Neural Nets Through Continuous Learning Rules},
  author={Irie, Kazuki and Faccio, Francesco and Schmidhuber, J{\"u}rgen},
  booktitle={Proc. Advances in Neural Information Processing Systems (NeurIPS)},
  address = {New Orleans, {LA}, {USA}},
  month = dec,
  year={2022}
}
```
