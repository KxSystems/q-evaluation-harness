# Morgan Stanley qqWen-72B-RL

**Provider**: Morgan Stanley
**Version**: v1.0
**Parameters**: 72B
**License**: Apache 2.0

## Evaluation Details
- **Dataset**: q-humaneval
- **Samples**: 50 per problem
- **Date**: 2025-10-22
- **Hardware**: 8xH100s

## Results Summary
- **Pass@1**: 45.10%
- **Pass@5**: 59.24%
- **Pass@10**: 62.63%

## Model Description
qqWen-72B-RL is a 72-billion parameter language model specifically designed for code generation in the Q programming language. Built upon the Qwen 2.5 architecture, this model has undergone a comprehensive training process including supervised fine-tuning (SFT) and reinforcement learning (RL) for the Q programming language. This is the largest model in the series and shows the best performance.

For more details, see our technical report: [https://arxiv.org/abs/2508.06813](https://arxiv.org/abs/2508.06813)

## Training Details
- **Base Model**: Qwen 2.5
- **Training Method**: Supervised Fine-Tuning (SFT) + Reinforcement Learning (RL)
- **Specialization**: Q programming language code generation
- **Total Solutions Evaluated**: 8,200
- **Passed Solutions**: 3,698
