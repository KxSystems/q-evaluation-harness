# Morgan Stanley qqWen-3B-RL

**Provider**: Morgan Stanley
**Version**: v1.0
**Parameters**: 3B
**License**: Apache 2.0

## Evaluation Details
- **Dataset**: q-humaneval
- **Samples**: 50 per problem
- **Date**: 2025-10-22
- **Hardware**: 8xH100s

## Results Summary
- **Pass@1**: 14.57%
- **Pass@5**: 25.92%
- **Pass@10**: 29.78%

## Model Description
qqWen-3B-RL is a 3-billion parameter language model specifically designed for code generation in the Q programming language. Built upon the Qwen 2.5 architecture, this model has undergone a comprehensive training process including supervised fine-tuning (SFT) and reinforcement learning (RL) for the Q programming language.

For more details, see our technical report: [https://arxiv.org/abs/2508.06813](https://arxiv.org/abs/2508.06813)

## Training Details
- **Base Model**: Qwen 2.5
- **Training Method**: Supervised Fine-Tuning (SFT) + Reinforcement Learning (RL)
- **Specialization**: Q programming language code generation
- **Total Solutions Evaluated**: 8,200
- **Passed Solutions**: 1,195
