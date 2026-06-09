# References Audit

> Checklist for verifying all citations in the GRL-Torus manuscript.
> The author should verify each item against the submitted paper.

---

## Citation Verification Checklist

### Completeness Issues

- [ ] **[12]** — Verify full author list, venue, year, and page numbers. Incomplete reference in the original submission.
- [ ] **[14]** — Add page numbers or article number if published in a journal.

### Misuse / Misattribution Issues

- [ ] **[11]** — Cited in the context of "IoT/battery" or "energy-efficient" routing. If GRL-Torus does NOT implement an energy model, this citation is misapplied. **Replace** with a citation that supports the actual claim being made (e.g., congestion-aware routing in data centre networks), or remove the claim.

### Required Additional References

The following references should be added to strengthen the Related Work and properly credit the techniques used:

| Technique Used | Recommended Citation | Status |
|----------------|---------------------|--------|
| GraphSAGE | Hamilton, W., Ying, R., & Leskovec, J. (2017). "Inductive representation learning on large graphs." *NeurIPS*. | [ ] Verify cited |
| Dueling DQN | Wang, Z., et al. (2016). "Dueling network architectures for deep reinforcement learning." *ICML*. | [ ] Verify cited |
| Prioritized Experience Replay | Schaul, T., et al. (2015). "Prioritized experience replay." *ICLR*. | [ ] Verify cited |
| Double DQN | van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double Q-learning." *AAAI*. | [ ] Verify cited |
| Odd-Even Turn Model | Chiu, G.-M. (2000). "The odd-even turn model for adaptive routing." *IEEE TPDS*, 11(7), 729-738. | [ ] Verify cited |
| Valiant Load Balancing | Valiant, L. G. (1982). "A scheme for fast parallel communication." *SIAM J. Comput.*, 11(2), 350-361. | [ ] Verify cited |
| XY Routing | Standard reference for dimension-order routing in mesh/torus networks. | [ ] Verify cited |
| SimPy | SimPy discrete-event simulation library — cite documentation or a paper using SimPy. | [ ] Verify cited |

### GNN-Routing Related Work to Compare Against

The reviewers requested comparison with at least 3 GNN-routing papers. Recommended additions:

| Paper | Relevance |
|-------|-----------|
| Almasan, P., et al. (2022). "Deep reinforcement learning meets graph neural networks: Exploring a routing optimization use case." *Computer Communications*, 196, 184-194. | GNN+DQN for SDN routing; closest comparator |
| Rusek, K., et al. (2020). "RouteNet: A graph neural network for network modeling and optimization in SDN." *IEEE JSAC*, 38(10), 2260-2270. | GNN for traffic engineering; different objective |
| Li, Y., et al. (2019). "Deep reinforcement learning for network routing." *NeurIPS Workshop on ML for Systems*. | RL for routing; no GNN |
| Mukhutdinov, D., et al. (2019). "Multi-agent reinforcement learning for packet routing in communication networks." *IEEE ICC*. | MARL for routing; scalability focus |

---

## Cross-Reference Verification

For each numbered citation `[n]` in the manuscript, verify:

1. **Existence**: The full bibliographic entry exists in the References section.
2. **Accuracy**: Author names, title, venue, year, and page numbers are correct.
3. **Relevance**: The citation supports the claim where it is used.
4. **Formatting**: Consistent style (IEEE or ACM).

### Specific Claims to Verify

| Manuscript Claim | Expected Citation | Action |
|-----------------|-------------------|--------|
| "2D torus widely adopted in HPC" | Should cite a systems paper (e.g., Cray Aries, IBM Blue Gene) | [ ] Verify |
| "GraphSAGE aggregation" | Hamilton et al. 2017 | [ ] Verify |
| "Dueling architecture" | Wang et al. 2016 | [ ] Verify |
| "Prioritized replay" | Schaul et al. 2015 | [ ] Verify |
| "Odd-Even routing prevents deadlock" | Chiu 2000 | [ ] Verify |
| "Valiant load balancing" | Valiant 1982 | [ ] Verify |

---

## Formatting Issues to Fix

- [ ] All author names use consistent format (First Last vs Last, F.)
- [ ] Journal titles are properly italicised
- [ ] Conference proceedings include location and publisher
- [ ] DOIs are included where available
- [ ] No duplicate references
- [ ] References are numbered sequentially without gaps
