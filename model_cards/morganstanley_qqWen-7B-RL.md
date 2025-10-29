# Morgan Stanley qqWen-7B-RL

**Provider**: Morgan Stanley
**Version**: v1.0
**Parameters**: 7B
**License**: Apache 2.0

## Evaluation Details
- **Dataset**: q-humaneval
- **Samples**: 50 per problem
- **Date**: 2025-10-22
- **Hardware**: 8xH100s

## Results Summary
- **Pass@1**: 24.63%
- **Pass@5**: 36.82%
- **Pass@10**: 40.84%

## Model Description
qqWen-7B-RL is a 7-billion parameter language model specifically designed for code generation in the Q programming language. Built upon the Qwen 2.5 architecture, this model has undergone a comprehensive training process including supervised fine-tuning (SFT) and reinforcement learning (RL) for the Q programming language.

For more details, see our technical report: [https://arxiv.org/abs/2508.06813](https://arxiv.org/abs/2508.06813)

## Training Details
- **Base Model**: Qwen 2.5
- **Training Method**: Supervised Fine-Tuning (SFT) + Reinforcement Learning (RL)
- **Specialization**: Q programming language code generation
- **Total Solutions Evaluated**: 8,200
- **Passed Solutions**: 2,020
