# Morgan Stanley qqWen-1.5B-SFT

**Provider**: Morgan Stanley
**Version**: v1.0
**Parameters**: 1.5B
**License**: Apache 2.0

## Evaluation Details
- **Dataset**: q-humaneval
- **Samples**: 50 per problem
- **Date**: 2025-10-22
- **Hardware**: 8xH100s

## Results Summary
- **Pass@1**: 7.01%
- **Pass@5**: 15.53%
- **Pass@10**: 19.91%

## Model Description
qqWen-1.5B-SFT is a 1.5-billion parameter language model specifically designed for code generation in the Q programming language. Built upon the Qwen 2.5 architecture, this model has undergone supervised fine-tuning (SFT) for the Q programming language.

For more details, see our technical report: [https://arxiv.org/abs/2508.06813](https://arxiv.org/abs/2508.06813)

## Training Details
- **Base Model**: Qwen 2.5
- **Training Method**: Supervised Fine-Tuning (SFT)
- **Specialization**: Q programming language code generation
- **Total Solutions Evaluated**: 8,200
- **Passed Solutions**: 575
