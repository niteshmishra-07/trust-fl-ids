# Team Workflow — Trust-Weighted FL for IoT IDS

## Work division

**Track A — Federated Core** (Nitesh) — branch `nitesh/federated-core`
- Phase 2: finish + verify `src/baseline_model.py` (centralized reference accuracy)
- Phase 3: finish + verify `src/fl_client.py` + `src/model_utils.py` (plain FedAvg working end-to-end)
- Phase 6: build `dashboard/app.py` (Streamlit) once Phase 3/5 numbers exist to visualize

**Track B — Trust & Robustness** (Chaitanya) — branch `chaitanya/trust-robustness`
- Phase 4: finish + verify `src/trust_strategy.py` (`TrustWeightedFedAvg`) — the core contribution
- Phase 5: attack simulation — use `poison_node()` in `src/common.py` + the `boost_factor` param
  already wired into `fl_client.py`'s `NodeClient.fit()` to simulate label-flip + model-replacement
  attacks. Produce the FedAvg-vs-Trust-under-attack comparison.
- Draft the Methodology + Results sections of the paper (Phase 7) from your own experiment output.

**Shared / integration** — `src/run_experiment.py`
Both strategies get wired together here for the final comparison. Don't develop new features
directly on this file in your own branch beyond what you need to test your own track — the final
version gets merged once both tracks are stable (see merge order below).

## Interface contract (already stable, don't change without telling the other person)

- `model_utils.get_weights(model)` / `set_weights(model, weights)` — how model params move in/out
  as plain lists of numpy arrays.
- `fl_client.NodeClient(node_id, node_df, scaler, boost_factor=1.0)` — one client per simulated node.
  `fit()` returns `(new_weights, num_examples, {"node_id": ...})`.
- Any Flower `Strategy` subclass (`FedAvg` or `TrustWeightedFedAvg`) just needs to implement
  `aggregate_fit(server_round, results, failures) -> (Parameters, metrics)`.

If either of you needs to change this contract, flag it to the other before merging — it breaks
the other track's code.

## Git workflow

1. Clone the repo, then check out your branch:
   ```
   git clone <repo-url>
   cd trust-fl-ids
   git checkout nitesh/federated-core        # or chaitanya/trust-robustness
   ```
2. Commit and push regularly to your own branch:
   ```
   git add -A
   git commit -m "Phase 3: FedAvg simulation converges to 85% on IID split"
   git push origin nitesh/federated-core
   ```
3. Open a Pull Request into `master` when your phase is verified working (include your
   accuracy/F1 numbers in the PR description). The other person reviews before merging —
   two sets of eyes on results going into the paper matters more than it does on typical code.
4. **Merge order:** Track A's Phase 3 should merge to `master` first (Track B's attack
   comparison script imports the FedAvg strategy as the control condition). After that,
   either track can merge independently.
5. Rebase on `master` periodically so you're not diverging for weeks:
   ```
   git fetch origin
   git rebase origin/master
   ```

## GitHub setup (one-time, whoever creates the repo)

```
# create an empty repo on github.com first, then from this local copy:
git remote add origin https://github.com/<your-org-or-username>/trust-fl-ids.git
git branch -M master main          # optional: rename to 'main' if you prefer
git push -u origin main
git push -u origin nitesh/federated-core
git push -u origin chaitanya/trust-robustness
```
Then add Chaitanya as a collaborator: repo Settings → Collaborators → Add people.

## Suggested checkpoints

| When | Sync point |
|---|---|
| End of this week | Track A: baseline + FedAvg both give sane, reproducible numbers on IID data |
| Next week | Track B: TrustWeightedFedAvg matches FedAvg on clean data (sanity check), starts attack testing |
| Week after | Merge both into `master`, run the full comparison suite together, start dashboard + write-up |
