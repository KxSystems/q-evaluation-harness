# Morgan Stanley qqWen-32B-RL-Reasoning

**Provider**: Morgan Stanley
**Version**: v1.0
**Parameters**: 32B
**License**: Apache 2.0

## Evaluation Details
- **Dataset**: q-humaneval
- **Samples**: 50 per problem
- **Date**: 2025-10-22
- **Hardware**: 8xH100s

## Results Summary
- **Pass@1**: 35.12%
- **Pass@5**: 48.76%
- **Pass@10**: 53.74%

## Model Description
qqWen-32B-RL is a 32-billion parameter language model specifically designed for advanced reasoning and code generation in the Q programming language. Built upon the Qwen 2.5 architecture, this model has undergone a comprehensive three-stage training process: pretraining, supervised fine-tuning (SFT), and reinforcement learning (RL) for the Q programming language. qqWen-32B-RL is a reasoning model.

For more details, see our technical report: [https://arxiv.org/abs/2508.06813](https://arxiv.org/abs/2508.06813)

## Training Details
- **Base Model**: Qwen 2.5
- **Training Method**: Pretraining + Supervised Fine-Tuning (SFT) + Reinforcement Learning (RL)
- **Specialization**: Q programming language code generation with reasoning
- **Total Solutions Evaluated**: 8,200
- **Passed Solutions**: 2,880
