
# Kaggle AI Mathematical Olympiad - Progress Prize 2 9th Place Solution (Fast-Math-R1-14B)
## Team members
- Hiroshi Yoshihara: Aillis Inc., Univ. of Tokyo
- Yuichi Inoue: Sakana AI
- Taiki Yamaguchi: Rist Inc.

# Usage
## 1. Set up environment
```bash
poetry lock
poetry install --no-root
```

## 2. Run first stage training
Training time: approx. 10 hours (8× H200 GPUs)
```bash
```

## 3. Run second stage training
Training time: approx. 16 hours (8× H200 GPUs)
```bash
```


# First stage: intensive SFT using a high-difficulty dataset
## Dataset
[OpenR1 Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k): We randomly sampled 3000 examples where the R1’s trace had more than 12800 tokens and an accuracy of over 50%, along with another 3000 examples where the accuracy ranged between 50% and 75%.
[openr1_hard](https://huggingface.co/datasets/hoanganhpham/openr1_hard):  "~2.5k hard samples from open-r1-math-220k. Samples deemed as hard were unsolvable by r1-distill-32b after 4 tries."
[Light-R1-SFTData](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData): We used the 2nd stage data from Light-R1-SFTData.

We merged all the datasets mentioned above, removed duplicates, and selected the correct generation with the shortest token length. For samples in the Light-R1 dataset where ground truth answers were not provided, we extracted and substituted the answers from the R1 traces. As a result, we constructed a **high-difficulty dataset consisting of 7900 problem - R1 trace - answer sets**.

## Training
Based on our prior experiments, we observed that the 14B models demonstrated more stable performance. Therefore, we chose DeepSeek-R1-Distill-Qwen-14B as our starting point. A full-parameter supervised fine-tuning training was conducted on a machine with 8 H200 GPUs, using the SFTTrainer from the trl library.

## Evaluation
We evaluated the model’s performance using a dataset of 40 problems, consisting of **10 reference problems and 30 from AIME 2025**.
Initially, we generated answers using 16k tokens × 32 prompts. We then applied post-hoc filtering based on token length and the number of prompts to assess performance under various inference conditions.

## Results

| Experiment                    | Token budget | Accuracy (majority@32) | Accuracy (pass@32) | Num of answers collected | Average generation length | Public LB (quantized model) |
|-------------------------------|---------------------|--------------------------|---------------------|--------------------------|---------------------------|-----------|
| DeepSeek-R1-Distill-Qwen-14B  | 16384               | 0.700                    | 0.775               | 21.775                   | 9684                      | 25        |
| DeepSeek-R1-Distill-Qwen-14B  | 12800               | 0.675                    | 0.775               | 16.775                   | 8331                      |           |
| DeepSeek-R1-Distill-Qwen-14B  | 9000                | 0.525                    | 0.600               | 12.500                   | 4725                      |           |
| SFT          | 16384               | 0.750                    | 0.825               | 20.725                   | 10396                     | 23        |
| SFT         | 12800               | 0.725                    | 0.725               | 15.700                   | 7024                      |           |
| SFT        | 9000                | 0.550                    | 0.550               | 11.600                   | 4387                      |           |


Our local validation scores improved remarkably after the first-stage SFT. However, we observed that the Public LB scores tended to be slightly lower.
We believe this was **due to increased reasoning redundancy introduced by SFT, causing many examples to fail to reach a conclusion within the time limit**.
To address this, our next objective was to apply reinforcement learning to encourage the model to reach accurate conclusions using fewer tokens, while maintaining performance.

# Second stage: GRPO for more efficient reasoning
## Dataset
[Light-R1-SFTData](https://huggingface.co/datasets/qihoo360/Light-R1-SFTData): We used the 2nd stage data from Light-R1-SFTData.

## Training
We used the [faster version of trl GRPOTrainer](https://github.com/nhannguyen2709/open-r1).

We used the following reward function:
1. Format reward
In our submission, generation is stopped at the `</think>` tag to save time, therefore we designed the reward to match the pattern `r"^.*?oxed{(.*?)}.*?</think>.*?$"`.
2. Cosine reward (correct [1.0, 0.1], incorrect [-0.1, -1.0], max_len=30000)
Compared to a normal accuracy-based reward, cosine reward applies a continuous penalty to longer correct reasoning traces and shorter incorrect ones.
3. Length reward
Length-based rewards to discourage overthinking and promote token efficiency.
Paper: https://arxiv.org/abs/2501.12599

## Evaluation
Same as first stage.

## Results
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F1973217%2Fdbd9dd1814ade77cc5c840319d36cf72%2FScreenshot%202025-04-02%20at%2016.55.07.png?generation=1743580526542546&alt=media)
The reward was optimized steadily throughout training, but after step 60, catastrophic shifts occurred, causing a substantial decline in performance. We thus decided to use checkpoints from earlier steps for evaluation.

| Experiment                    | Token budget | Accuracy (majority@32) | Accuracy (pass@32) | Num of answers collected | Average generation length | Public LB (quantized model) |
|-------------------------------|---------------------|--------------------------|---------------------|--------------------------|---------------------------|-----------|
| DeepSeek-R1-Distill-Qwen-14B  | 12800               | 0.675                    | 0.775               | 16.775                   | 8331                      | 25        |
| DeepSeek-R1-Distill-Qwen-14B  | 9000                | 0.525                    | 0.600               | 12.5                     | 4725                      |           |
| SFT         | 12800               | 0.725                    | 0.725               | 15.7                     | 7024                      | 23        |
| SFT         | 9000                | 0.550                    | 0.550               | 11.6                     | 4387                      |           |
| SFT + GRPO (best checkpoint)       | 12800               | **0.725**                | **0.775**           | **18.5**               | 6817                      | **29**    |
| SFT + GRPO (best checkpoint)       | 9000                | **0.625**                | **0.700**           | **15.25**                | 4759                      |           |

GRPO enabled us to train a model that preserved accuracy while significantly improving inference efficiency through shorter token lengths. 
