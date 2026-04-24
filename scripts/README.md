# Scripts By Paper Workflow

| Paper component | Script | Purpose |
| --- | --- | --- |
| Section 4 main results | `run_encode_index_groups.sh` | Build full BGE-M3 mMARCO document indexes. |
| Section 4 main results | `run_all_vector_pairs.sh` | Run the vector interpolation matrix. |
| Section 4.1 / Appendix A | `generate_word_mix.sh` | Generate word-level code-mixed query bands. |
| Section 4.1 / Appendix B | `reproduce_word_mix.sh` | Run word-mix validation and matching embedding-mix retrieval on 100k subsets. |
| Section 5 ablation | `run_encode_index_ablation.sh` | Build ablation indexes. |
| Section 5 ablation | `run_ablation.sh` | Run model-family and scale ablation jobs. |

Most scripts expose paths and hardware placement through environment variables. See `docs/SCRIPTS.md` for the longer variable map.
