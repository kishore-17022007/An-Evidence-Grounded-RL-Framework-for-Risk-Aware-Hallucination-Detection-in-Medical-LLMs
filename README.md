# MedVerify-RL

### Evidence-Grounded Reinforcement Learning for Risk-Aware Hallucination Detection in Medical LLMs

MedVerify-RL is a research prototype exploring Deep Q-Networks (DQN) and Hierarchical Reinforcement Learning (HRL) for evidence-informed assessment of medical claims. It combines a MedHallu-based simulation environment with a separate Europe PMC literature-verification workflow.

## Algorithms Used

| Algorithm / Method | Purpose |
|---|---|
| **Deep Q-Network (DQN)** | Learns verification actions using online and target Q-networks, experience replay, and epsilon-greedy exploration. |
| **Hierarchical Reinforcement Learning (HRL)** | Uses a manager and four specialized DQN workers to select and execute verification options. |
| **Semi-Markov Decision Process (SMDP)** | Enables the HRL manager to learn from temporally extended actions and discounted option returns. |
| **Experience Replay** | Reuses stored transitions to train the agents. |
| **Target Network Updates** | Improves training stability through hard or soft updates. |
| **Deterministic Evidence Ranking** | Ranks relevant abstract excerpts using lexical heuristics in the separate verification workflow. |

## Verification Actions

The simulated agents operate in a Gymnasium environment with a 15-dimensional observation space and eight discrete actions:

- Retrieve evidence
- Verify the current claim
- Investigate contradictions
- Request an independent source
- Finalize as supported
- Finalize as contradicted
- Finalize as insufficient evidence
- Abstain or escalate

The HRL manager selects among four options: evidence gathering, contradiction investigation, verification assessment, and finalization.

## Experimental Results

The following results come from the documented 10,000-interaction pilot experiments, using seed 17 and 186 paired validation scenarios.

| Metric | DQN | HRL |
|---|---:|---:|
| Training interactions | 10,000 | 10,000 |
| Training episodes | 7,113 | 7,682 |
| Mean validation target-free return | -0.1426 | -0.0257 |
| Mean validation episode length | 3.6075 | 1.0269 |
| Validation assessment coverage | 100% | 100% |
| Validation abstention rate | 100% | 100% |
| Answer-origin alignment | 0% | 0% |

**Interpretation:** HRL achieved a less negative mean target-free return and shorter validation episodes in this pilot. However, both agents abstained in every validation episode, so these results do not demonstrate successful hallucination detection or the superiority of HRL. The zero answer-origin alignment indicates that neither agent's final predictions matched the selected answer-origin targets in the evaluated scenarios.

## Research and Evaluation

- Paired validation evaluation using the same ordered scenarios and reset-seed schedule.
- Fixed-final-checkpoint protocol for reproducibility.
- Training and validation split separation, with no test-split use during training or checkpoint selection.
- Action-frequency, episode-length, reward, and HRL option-behavior analysis.
- A curated collection of 15 research figures.
- Automated software tests for implementation and workflow checks.

## Evidence Verification

The separate verification service uses the **Europe PMC REST API** to retrieve publication metadata and abstracts. It applies deterministic sentence segmentation, lexical ranking, excerpt provenance tracking, and structured claim assessment.

It can report insufficient evidence when available excerpts do not satisfy its baseline rules. However, lexical similarity is not semantic entailment, and retrieved abstracts do not establish medical truth. The live verification service is not yet integrated with the trained RL policies.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| RL environment | Gymnasium |
| Dataset | MedHallu |
| Literature retrieval | Europe PMC REST API |
| Dashboard | Streamlit |
| Testing | pytest |
| Visualization | Matplotlib-based research figures |

## Limitations

- MedHallu answer-origin alignment is not medical accuracy.
- Simulator evidence signals do not establish clinical evidence quality.
- The literature workflow uses lexical heuristics rather than a semantic entailment model.
- Live Europe PMC evidence is not consumed by the current DQN or HRL agents.
- Clinical validation has not been performed.

## Future Work

- Integrate retrieved evidence into the RL observation space.
- Explore MAPPO and alternative reinforcement learning algorithms.
- Incorporate semantic entailment and claim-level evidence assessment.
- Improve source independence, uncertainty estimation, and abstention behavior.
- Evaluate against independently annotated claim-level benchmarks.
